import gzip
import logging
import urllib.request
from pathlib import Path
from typing import Any, Literal

import numpy as np
from tqdm.auto import tqdm

from lab.utils.io import get_root, read_json, write_json

from .base import BaseDataset

# Mirror of the original files from https://yann.lecun.com/exdb/mnist/
_MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
_MNIST_FILES = {
    'train': ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
    'test': ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'),
}

logger = logging.getLogger(__name__)


class MNISTDataset(BaseDataset):
    """
    MNIST dataset, see https://yann.lecun.com/exdb/mnist/.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        split: Literal['train', 'test'] = 'train',
        limit: int | None = None,
        shuffle_index: bool = False,
        force_reindex: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        root : Path or str or None, optional
            Dataset directory relative to the project root. If None, 'data/mnist/{split}'.
        split : {'train', 'test'}, optional
            Dataset split.
        limit : int or None, optional
            Limit the number of samples.
        shuffle_index : bool, optional
            Whether to shuffle the index.
        force_reindex : bool, optional
            Recreate the index file even if it exists.
        """
        if root is None:
            root = get_root() / 'data' / 'mnist' / split
        else:
            root = get_root() / root

        index_path = root / 'index.json'
        if index_path.exists() and not force_reindex:
            index: list[dict[str, Any]] = read_json(str(index_path))  # ty:ignore[invalid-assignment]
        else:
            index: list[dict[str, Any]] = self._create_index(split, root)

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
        )

    def _create_index(
        self,
        split: Literal['train', 'test'],
        data_path: Path,
    ) -> list[dict[str, Any]]:
        """
        Download MNIST, save each image as a .npy file and write index.json.

        Returns
        -------
        list[dict[str, Any]]
            Index with the 'path' and 'label' of each element.
        """

        index: list[dict[str, Any]] = []
        data_path.mkdir(exist_ok=True, parents=True)

        images, labels = self._download(split, data_path.parent / 'raw')

        logger.info(f'Processing MNIST data for {split} part...')
        for i, (img, label) in tqdm(enumerate(zip(images, labels, strict=True)), total=len(images)):
            img = (img / 255.0).astype(np.float32)

            save_path = data_path / f'{i:06}.npy'
            np.save(save_path, img)

            index.append({'path': str(save_path), 'label': int(label)})

        write_json(index, str(data_path / 'index.json'))

        logger.info(f'Successfully processed {len(index)} {split} MNIST images.')
        return index

    @staticmethod
    def _download(split: Literal['train', 'test'], raw_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Download the IDX files of the split (if not cached in 'raw_path') and parse them.

        Returns
        -------
        images : np.ndarray
            uint8 array of shape (N, 28, 28).
        labels : np.ndarray
            uint8 array of shape (N,).
        """
        raw_path.mkdir(exist_ok=True, parents=True)
        arrays = []
        for filename in _MNIST_FILES[split]:
            file_path = raw_path / filename
            if not file_path.exists():
                logger.info(f'Downloading {_MNIST_URL + filename}...')
                urllib.request.urlretrieve(_MNIST_URL + filename, file_path)
            with gzip.open(file_path, 'rb') as f:
                data = f.read()
            # IDX header: magic number (its last byte is the number of dims), then dim sizes
            ndim = data[3]
            shape = np.frombuffer(data, dtype='>u4', count=ndim, offset=4)
            arrays.append(np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape))
        images, labels = arrays
        return images, labels
