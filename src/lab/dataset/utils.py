import grain
from absl import flags
from omegaconf import DictConfig


def build_dataloader(
    config: DictConfig,
    dataset: grain.sources.RandomAccessDataSource,
    train: bool,
    batch_size: int | None = None,
) -> grain.IterDataset:
    """
    Build the dataloader of one partition (shuffling, batching, threaded reading and
    optional multiprocess prefetching).

    Parameters
    ----------
    config : DictConfig
        Partition dataloader config.
    dataset : grain.sources.RandomAccessDataSource
        Dataset of the partition.
    train : bool
        Repeat the data infinitely (train) or pass it once (eval).
    batch_size : int or None, optional
        Overrides 'config.batch_size'.

    Returns
    -------
    grain.IterDataset
        Dataloader of batches.
    """
    ds = grain.MapDataset.source(dataset)
    if config.shuffle:
        ds = ds.shuffle(seed=config.seed)

    if train:
        ds = ds.repeat()

    # Batching a MapDataset (by index) is much faster than batching the examples of an
    # IterDataset one by one; it is possible only without 'filter' in the pipeline, and a
    # filter would need the batching after 'to_iter_dataset'
    ds = ds.batch(batch_size or config.batch_size, drop_remainder=config.drop_remainder)
    ds = ds.to_iter_dataset(
        grain.ReadOptions(
            num_threads=config.get('num_threads', 1),
            # In batches (not examples)
            prefetch_buffer_size=config.get('prefetch_buffer_size', 8),
        )
    )

    if (num_workers := config.get('num_workers', 0)) > 0:
        # The grain workers read the absl flags, which Hydra does not parse
        if not flags.FLAGS.is_parsed():
            flags.FLAGS.mark_as_parsed()
        ds = ds.mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=num_workers,
                per_worker_buffer_size=config.get('per_worker_buffer_size', 1),
            )
        )
    return ds


def get_dataloaders(
    config: DictConfig,
    datasets: dict[str, grain.sources.RandomAccessDataSource],
    grad_accum_steps: int = 1,
) -> dict[str, grain.IterDataset]:
    """
    Create dataloaders for the 'train', 'val' and 'test' partitions that have a config
    under 'config.dataloader'.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    datasets : dict[str, grain.sources.RandomAccessDataSource]
        Dataset of each partition.
    grad_accum_steps : int, optional
        The train batch size is multiplied by it (a batch holds all micro-batches of a step).

    Returns
    -------
    dict[str, grain.IterDataset]
        Dataloader of each partition.
    """
    dataloaders = {}
    for partition, dataset in datasets.items():
        if (
            partition not in ['train', 'val', 'test']
            or (loader_cfg := config.dataloader.get(partition)) is None
        ):
            continue

        dataloaders[partition] = build_dataloader(
            loader_cfg,
            dataset,
            train=partition == 'train',
            batch_size=(loader_cfg.batch_size * grad_accum_steps if partition == 'train' else None),
        )
    return dataloaders
