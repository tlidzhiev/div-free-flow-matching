"""Generate Kolmogorov-flow velocity fields for Re-conditional flow matching (NumPy/SciPy).

Each trajectory: random band-limited vorticity -> spin-up onto the attractor -> snapshots
every `--t-snap`. Solved with `lab.dataset.solver` on a `--sim-n`^2 grid, then spectrally
truncated to `--n`^2. Velocity is recomputed from the truncated vorticity, so the stored
fields are divergence-free on the output grid (up to float64 round-off).

Everything is computed and stored in float64; cast to float32 in the dataset/training code.

Parallelism: trajectories are distributed over `--processes` worker processes, each runs
`--batch` trajectories at a time with `--fft-workers` FFT threads. On CPU, many processes
with single-threaded FFTs give the best throughput. The initial condition of trajectory `i`
of a split depends only on (`--seed`, split, i), so the data does not depend on the
parallelization settings.

Condition: c = normalized log(nu) in [-1, 1] over the train range [re_min, re_max]
(c = -1 at re_max, c = +1 at re_min).

Splits are made by trajectory (no snapshot of one trajectory is in two splits):
  * train: Re log-uniform (stratified) in [re_min, re_max], one Re per trajectory;
  * val / test: Re on the fixed grid `--eval-re`, several trajectories per value
    (convenient for per-Re statistics of generated samples).

Output (`--out`):
    meta.json                      arguments, time step, normalization, solver constants
    {split}/fields.npy  float64    (N, n, n, 2)  velocity (u, v), indexing [x, y], channels-last
    {split}/c.npy       float64    (N,)          normalized log(nu)
    {split}/re.npy      float64    (N,)
    {split}/nu.npy      float64    (N,)
    {split}/traj.npy    int32      (N,)          trajectory id within the split
    {split}/time.npy    float64    (N,)          time since the end of the spin-up

`fields.npy` can be opened with `np.load(path, mmap_mode='r')`.

Examples:
    uv run scripts/generate_data.py --out data/kolmogorov
    # fast smoke test
    uv run scripts/generate_data.py --out data/kolmogorov_debug --sim-n 128 \
        --train-traj 4 --val-traj-per-re 1 --test-traj-per-re 1 --n-snapshots 4 --spinup 5
"""

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import scipy.fft
from numpy.lib.format import open_memmap
from tqdm import tqdm

from lab.dataset.solver import solver

DTYPE = np.float64
SPLITS = ('train', 'val', 'test')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--out', type=Path, default=Path('data/kolmogorov'))
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    p.add_argument('--seed', type=int, default=0)
    # grids
    p.add_argument('--sim-n', type=int, default=256, help='simulation grid')
    p.add_argument('--n', type=int, default=64, help='output grid (spectral truncation)')
    # physics
    p.add_argument('--re-min', type=float, default=100.0)
    p.add_argument('--re-max', type=float, default=1000.0)
    p.add_argument('--alpha', type=float, default=0.1, help='linear drag')
    p.add_argument('--k-f', type=int, default=4, help='forcing wavenumber')
    # time
    p.add_argument('--spinup', type=float, default=20.0, help='time to reach the attractor')
    p.add_argument('--t-snap', type=float, default=1.0, help='time between snapshots')
    p.add_argument('--n-snapshots', type=int, default=32, help='snapshots per trajectory')
    p.add_argument('--u-max', type=float, default=8.0, help='velocity bound for the time step')
    p.add_argument('--cfl', type=float, default=0.5)
    # dataset size
    p.add_argument('--train-traj', type=int, default=256)
    p.add_argument('--eval-re', type=float, nargs='+', default=[100.0, 200.0, 500.0, 1000.0])
    p.add_argument('--val-traj-per-re', type=int, default=8)
    p.add_argument('--test-traj-per-re', type=int, default=16)
    # parallelism
    p.add_argument('--processes', type=int, default=os.cpu_count(), help='worker processes')
    p.add_argument('--batch', type=int, default=1, help='trajectories per worker task')
    p.add_argument('--fft-workers', type=int, default=1, help='FFT threads per process')
    # checks
    p.add_argument(
        '--max-actual-cfl', type=float, default=1.0, help='abort if exceeded (RK3 limit ~1.7)'
    )
    p.add_argument(
        '--max-tail', type=float, default=1e-3, help='warn if the enstrophy tail exceeds this'
    )
    args = p.parse_args()

    assert args.n <= args.sim_n and args.n % 2 == 0, '--n must be even and <= --sim-n'
    assert 0 < args.re_min <= args.re_max
    for re in args.eval_re:
        assert args.re_min <= re <= args.re_max, f'eval Re={re} is outside the train range'
    return args


# ----------------------------------------------------------------------------- setup


def time_grid(args: argparse.Namespace) -> tuple[float, int, int]:
    """Common dt for all trajectories: t_snap is an exact multiple of dt."""
    dt_max = solver.stable_time_step(args.u_max, args.sim_n, args.cfl)
    steps_per_snapshot = math.ceil(args.t_snap / dt_max - 1e-9)
    dt = args.t_snap / steps_per_snapshot
    spinup_steps = max(1, math.ceil(args.spinup / dt - 1e-9))
    return dt, steps_per_snapshot, spinup_steps


def sample_re(split: str, args: argparse.Namespace) -> np.ndarray:
    """Reynolds number of every trajectory of the split."""
    if split == 'train':
        # stratified log-uniform: even coverage of [re_min, re_max]
        rng = np.random.default_rng(args.seed)
        m = args.train_traj
        u = (np.arange(m) + rng.uniform(size=m)) / m
        log_re = np.log(args.re_min) + u * (np.log(args.re_max) - np.log(args.re_min))
        return rng.permutation(np.exp(log_re))
    per_re = args.val_traj_per_re if split == 'val' else args.test_traj_per_re
    return np.repeat(np.asarray(args.eval_re, dtype=np.float64), per_re)


def normalize_log_nu(nu: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    lo, hi = np.log(1 / args.re_max), np.log(1 / args.re_min)
    if hi == lo:
        return np.zeros_like(nu)
    return 2 * (np.log(nu) - lo) / (hi - lo) - 1


# ----------------------------------------------------------------------------- fields


def truncate(w: np.ndarray, n: int) -> np.ndarray:
    """Spectral truncation (..., N, N) -> (..., n, n), keeping |k_x|, |k_y| < n / 2."""
    big = w.shape[-1]
    if big == n:
        return w
    w_hat = scipy.fft.fft2(w, workers=solver.FFT_WORKERS)
    k = np.fft.fftfreq(n, 1 / n).astype(np.int64)
    idx = k % big
    keep = np.abs(k) < n // 2  # drop the Nyquist modes of the target grid
    sub = w_hat[..., idx[:, None], idx[None, :]] * (keep[:, None] & keep[None, :])
    return scipy.fft.ifft2(sub, workers=solver.FFT_WORKERS).real * (n / big) ** 2


def enstrophy_tail(w: np.ndarray) -> np.ndarray:
    """Fraction of enstrophy near the dealiasing cutoff of the simulation grid, (..., N, N) -> (...).

    Large values mean the simulation is under-resolved for this Re."""
    g = solver.SpectralGrid.create(w.shape[-1], w.dtype)
    e = np.abs(scipy.fft.rfft2(w, workers=solver.FFT_WORKERS)) ** 2
    return (e * (np.sqrt(g.k2) > 0.8 * g.n / 3)).sum(axis=(-2, -1)) / e.sum(axis=(-2, -1))


@dataclass(frozen=True)
class RunConfig:
    seed: int
    split_index: int
    sim_n: int
    n: int
    alpha: float
    k_f: int
    dt: float
    steps_per_snapshot: int
    spinup_steps: int
    n_snapshots: int


def run_trajectories(
    cfg: RunConfig, task: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trajectories `ids` with viscosities `nu` -> uv (b, T, n, n, 2), speed, tail, rel_div (b,)."""
    ids, nu = task
    w0 = np.stack(
        [
            solver.random_vorticity(
                np.random.default_rng([cfg.seed, cfg.split_index, i]), cfg.sim_n
            )
            for i in ids
        ]
    ).astype(DTYPE)
    run = partial(solver.simulate, nu=nu, dt=cfg.dt, alpha=cfg.alpha, k_f=cfg.k_f)
    w0 = run(w0, n_snapshots=1, steps_per_snapshot=cfg.spinup_steps)[:, -1]
    traj = run(w0, n_snapshots=cfg.n_snapshots, steps_per_snapshot=cfg.steps_per_snapshot)

    speed = np.sqrt((solver.vorticity_to_velocity(traj) ** 2).sum(axis=-3)).max(axis=(1, 2, 3))
    tail = enstrophy_tail(traj).max(axis=1)

    w = truncate(traj, cfg.n)  # (b, T, n, n)
    uv = solver.vorticity_to_velocity(w)  # (b, T, 2, n, n)
    div = solver.divergence(uv)
    rel_div = np.sqrt((div**2).mean(axis=(1, 2, 3)) / (w**2).mean(axis=(1, 2, 3)))
    return np.moveaxis(uv, 2, -1), speed, tail, rel_div


def _init_worker(fft_workers: int) -> None:
    solver.FFT_WORKERS = fft_workers


# ----------------------------------------------------------------------------- main


def generate_split(
    split: str, args: argparse.Namespace, pool: Pool, dt: float, steps: int, spinup_steps: int
) -> dict:
    re = sample_re(split, args)
    nu = 1 / re
    m, t, n = len(re), args.n_snapshots, args.n
    out = args.out / split
    out.mkdir(parents=True)

    cfg = RunConfig(
        seed=args.seed,
        split_index=SPLITS.index(split),
        sim_n=args.sim_n,
        n=n,
        alpha=args.alpha,
        k_f=args.k_f,
        dt=dt,
        steps_per_snapshot=steps,
        spinup_steps=spinup_steps,
        n_snapshots=t,
    )
    starts = range(0, m, args.batch)
    tasks = [(np.arange(s, min(s + args.batch, m)), nu[s : s + args.batch]) for s in starts]

    fields = open_memmap(out / 'fields.npy', mode='w+', dtype=DTYPE, shape=(m * t, n, n, 2))
    speed, tail, rel_div = np.zeros(m), np.zeros(m), np.zeros(m)
    results = pool.imap(partial(run_trajectories, cfg), tasks)
    with tqdm(total=m, desc=split) as bar:
        for start, (uv, s, tl, rd) in zip(starts, results):
            stop = start + len(uv)
            assert uv.dtype == DTYPE, f'expected {DTYPE.__name__}, got {uv.dtype}'
            if not np.isfinite(uv).all():
                raise FloatingPointError(f'{split}: NaN/inf in trajectories {start}..{stop - 1}')
            fields[start * t : stop * t] = uv.reshape(-1, n, n, 2)
            speed[start:stop], tail[start:stop], rel_div[start:stop] = s, tl, rd
            bar.update(len(uv))
    fields.flush()

    per_sample = lambda x: np.repeat(x, t)
    np.save(out / 're.npy', per_sample(re).astype(DTYPE))
    np.save(out / 'nu.npy', per_sample(nu).astype(DTYPE))
    np.save(out / 'c.npy', per_sample(normalize_log_nu(nu, args)).astype(DTYPE))
    np.save(out / 'traj.npy', np.repeat(np.arange(m, dtype=np.int32), t))
    np.save(out / 'time.npy', np.tile(args.t_snap * np.arange(1, t + 1, dtype=DTYPE), m))

    cfl = speed * dt / (2 * np.pi / args.sim_n)
    if cfl.max() > args.max_actual_cfl:
        raise RuntimeError(
            f'{split}: actual CFL {cfl.max():.2f} > {args.max_actual_cfl} '
            f'(max|u| = {speed.max():.2f}); increase --u-max'
        )
    bad = tail > args.max_tail
    if bad.any():
        print(
            f'WARNING {split}: enstrophy tail up to {tail.max():.1e} for Re in '
            f'[{re[bad].min():.0f}, {re[bad].max():.0f}]: under-resolved, increase --sim-n'
        )

    fields = np.load(out / 'fields.npy', mmap_mode='r')
    return {
        'num_trajectories': m,
        'num_samples': m * t,
        'max_speed': float(speed.max()),
        'max_cfl': float(cfl.max()),
        'max_enstrophy_tail': float(tail.max()),
        'max_relative_divergence': float(rel_div.max()),
        'u_rms': float(np.sqrt(np.mean(np.square(fields, dtype=np.float64)))),
    }


def main() -> None:
    args = parse_args()
    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f'{args.out} exists; pass --overwrite')
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    dt, steps, spinup_steps = time_grid(args)
    print(
        f'processes={args.processes}x{args.fft_workers} threads  batch={args.batch}  '
        f'dtype={DTYPE.__name__}  sim {args.sim_n}^2 -> {args.n}^2  '
        f'Re in [{args.re_min:g}, {args.re_max:g}]  dt={dt:.3e}  '
        f'steps/snapshot={steps}  spin-up steps={spinup_steps}'
    )

    stats = {}
    with Pool(args.processes, initializer=_init_worker, initargs=(args.fft_workers,)) as pool:
        for split in args.splits:
            t0 = time.perf_counter()
            stats[split] = generate_split(split, args, pool, dt, steps, spinup_steps)
            stats[split]['seconds'] = round(time.perf_counter() - t0, 1)
            print(f'{split}: {stats[split]}')

    meta = {
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'solver': 'numpy/scipy',
        'equation': 'd_t w + u.grad w = nu lap w - alpha w - k_f cos(k_f y), domain [0, 2pi)^2',
        'layout': 'fields[i, x, y, (u, v)], indexing ij, x = 2 pi i / n',
        'dt': dt,
        'steps_per_snapshot': steps,
        'spinup_steps': spinup_steps,
        'condition': {
            'name': 'c',
            'definition': '2 (log nu - log nu_min) / (log nu_max - log nu_min) - 1',
            'log_nu_min': math.log(1 / args.re_max),
            'log_nu_max': math.log(1 / args.re_min),
        },
        # one global scale for both components keeps the fields divergence-free
        'normalization': {'u_scale': stats.get('train', {}).get('u_rms')},
        'splits': stats,
    }
    with (args.out / 'meta.json').open('w') as f:
        json.dump(meta, f, indent=4)
    print(f'saved to {args.out}')


if __name__ == '__main__':
    main()
