"""Generate Kolmogorov-flow velocity fields for Re-conditional flow matching (JAX, float64).

Each trajectory: random band-limited vorticity -> spin-up onto the attractor -> snapshots
every `--t-snap`. Solved with `lab.dataset.solver` on a `--sim-n`^2 grid, then
spectrally truncated to `--n`^2. Velocity is recomputed from the truncated vorticity, so the stored
fields are divergence-free on the output grid (up to float64 round-off).

Everything is computed and stored in float64 (JAX with x64 enabled); cast to float32 in the
dataset/training code. TPUs have no float64 FFTs: run on a GPU (best with fast float64, e.g.
A100/H100) or on the CPU (slower than on a GPU).

Parallelism: `--batch` trajectories are simulated at once on the device, as one compiled
program, together with their diagnostics and the truncation to the output grid; the next
batch is dispatched before the previous one is copied to the host and written, so the device
does not wait for the host. The initial noise of trajectory `i` of a split depends only on
(`--seed`, split, i) and is drawn with NumPy, so the data does not depend on the batch size
or the device (up to float64 round-off, ~1e-13 relative).

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
import shutil
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from lab.dataset.solver import solver

# float64 in JAX: before any JAX array is created (otherwise JAX silently computes in float32)
jax.config.update('jax_enable_x64', True)

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
    # device
    p.add_argument(
        '--device', default='auto', help='"auto" or a JAX platform ("gpu", "cpu"); no TPU'
    )
    p.add_argument(
        '--batch',
        type=int,
        default=32,
        help='trajectories simulated at once; peak device memory ~7 x batch x n-snapshots x '
        'sim-n^2 x 8 B (~3.5 GB for the defaults)',
    )
    # checks
    p.add_argument(
        '--max-actual-cfl', type=float, default=1.0, help='abort if exceeded (RK3 limit ~1.7)'
    )
    p.add_argument(
        '--max-tail', type=float, default=1e-3, help='warn if the enstrophy tail exceeds this'
    )
    args = p.parse_args()

    assert args.n <= args.sim_n and args.n % 2 == 0, '--n must be even and <= --sim-n'
    assert args.batch >= 1, '--batch must be positive'
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


@jax.jit(static_argnames=('n',))
def analyze(traj: jax.Array, n: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Vorticity (b, T, N, N) -> uv (b, T, n, n, 2), speed, tail, rel_div (b,).

    Velocity is recomputed from the vorticity truncated to the output grid, so the stored
    fields are divergence-free on it; speed (for the CFL check) and the enstrophy tail are
    taken on the simulation grid."""
    speed = jnp.sqrt((solver.vorticity_to_velocity(traj) ** 2).sum(axis=-3)).max(axis=(1, 2, 3))
    tail = solver.enstrophy_tail(traj).max(axis=1)

    w = solver.truncate(traj, n)  # (b, T, n, n)
    uv = solver.vorticity_to_velocity(w)  # (b, T, 2, n, n)
    div = solver.divergence(uv)
    rel_div = jnp.sqrt((div**2).mean(axis=(1, 2, 3)) / (w**2).mean(axis=(1, 2, 3)))
    return jnp.moveaxis(uv, 2, -1), speed, tail, rel_div


def launch_trajectories(
    cfg: RunConfig,
    ids: np.ndarray,
    nu: np.ndarray,
    device: jax.Device,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Dispatch the trajectories `ids` with viscosities `nu` on the device: the simulation and
    the diagnostics (see 'analyze'), without waiting for them.

    The initial noise of trajectory `i` comes from a NumPy generator seeded with
    (seed, split, i), so it does not depend on the batch or the device."""
    noise = np.stack(
        [
            np.random.default_rng([cfg.seed, cfg.split_index, i]).standard_normal(
                (cfg.sim_n, cfg.sim_n)
            )
            for i in ids
        ]
    ).astype(DTYPE)
    noise, nu = jax.device_put((noise, nu.astype(DTYPE)), device)
    run = partial(solver.simulate, nu=nu, dt=cfg.dt, alpha=cfg.alpha, k_f=cfg.k_f)
    w0 = run(solver.random_vorticity(noise), n_snapshots=1, steps_per_snapshot=cfg.spinup_steps)
    traj = run(w0[:, -1], n_snapshots=cfg.n_snapshots, steps_per_snapshot=cfg.steps_per_snapshot)
    return analyze(traj, cfg.n)


# ----------------------------------------------------------------------------- main


def generate_split(
    split: str,
    args: argparse.Namespace,
    device: jax.Device,
    dt: float,
    steps: int,
    spinup_steps: int,
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
    starts = list(range(0, m, args.batch))

    fields = open_memmap(out / 'fields.npy', mode='w+', dtype=DTYPE, shape=(m * t, n, n, 2))
    speed, tail, rel_div = np.zeros(m), np.zeros(m), np.zeros(m)

    def launch(start: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        stop = min(start + args.batch, m)
        return launch_trajectories(cfg, np.arange(start, stop), nu[start:stop], device)

    with tqdm(total=m, desc=split) as bar:
        pending = launch(starts[0])
        for index, start in enumerate(starts):
            current = pending
            # The next batch runs on the device while the host writes the current one
            if index + 1 < len(starts):
                pending = launch(starts[index + 1])
            uv, s, tl, rd = (np.asarray(x) for x in jax.device_get(current))
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

    device = jax.devices()[0] if args.device == 'auto' else jax.devices(args.device)[0]
    if device.platform == 'tpu':
        raise RuntimeError('TPUs have no float64 FFTs: run on a GPU or the CPU (--device)')
    if device.platform == 'cpu' and args.device == 'auto':
        print(
            'WARNING: no GPU found, running on the CPU (slow). For a GPU install JAX with CUDA, '
            "e.g. uv pip install 'jax[cuda12]==<the jax version of pyproject.toml>'"
        )

    dt, steps, spinup_steps = time_grid(args)
    print(
        f'device={device}  batch={args.batch}  '
        f'dtype={DTYPE.__name__}  sim {args.sim_n}^2 -> {args.n}^2  '
        f'Re in [{args.re_min:g}, {args.re_max:g}]  dt={dt:.3e}  '
        f'steps/snapshot={steps}  spin-up steps={spinup_steps}'
    )

    stats = {}
    for split in args.splits:
        t0 = time.perf_counter()
        stats[split] = generate_split(split, args, device, dt, steps, spinup_steps)
        stats[split]['seconds'] = round(time.perf_counter() - t0, 1)
        print(f'{split}: {stats[split]}')

    meta = {
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'solver': f'jax ({device.platform}, {DTYPE.__name__})',
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
