import random
from typing import Any

import grain
import numpy as np


class BaseDataset(grain.sources.RandomAccessDataSource):
    """
    Base class for the datasets.

    Given an index (list of dicts), processes different datasets for the same
    task in the identical manner. A subclass only has to build the index.
    """

    def __init__(
        self,
        index: list[dict[str, Any]],
        limit: int | None = None,
        shuffle_index: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        index : list[dict[str, Any]]
            Metadata of each element; the 'path' field is required.
        limit : int or None, optional
            If not None, keep only the first 'limit' elements.
        shuffle_index : bool, optional
            Shuffle the index with a fixed seed (42).
        """
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: list[dict[str, Any]] = index

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Load an element and combine it into a dict.

        The key names ('img', 'label') must stay consistent across the dataset,
        collate_fn, loss and model.
        """
        data_dict = self._index[index]
        data_path = data_dict['path']
        data_object = self.load_object(data_path)
        return {'img': data_object, 'label': np.int64(data_dict['label'])}

    def __len__(self) -> int:
        return len(self._index)

    def load_object(self, path: str) -> np.ndarray:
        """
        Load an array from disk.
        """
        return np.load(path)

    @staticmethod
    def _filter_records_from_dataset(
        index: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Filter elements of the index.

        Not used yet; call before shuffling and limiting.
        """
        # Filter logic
        return index

    @staticmethod
    def _assert_index_is_valid(index: list[dict[str, Any]]) -> None:
        """
        Check that every index entry has the 'path' field.
        """
        for entry in index:
            assert 'path' in entry, (
                "Each dataset item should include field 'path' - path to sample file."
            )

    @staticmethod
    def _sort_index(index: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Sort the index.

        Not used yet; call after filtering and before shuffling and limiting.
        """
        # Sorting logic
        return index

    @staticmethod
    def _shuffle_and_limit_index(
        index: list[dict[str, Any]],
        limit: int | None,
        shuffle_index: bool,
    ) -> list[dict[str, Any]]:
        """
        Shuffle the index with a fixed seed and limit its length.
        """
        if shuffle_index:
            # Use a local Random instance so this doesn't reset the global
            # RNG state (and clobber the configured trainer/inferencer seed).
            random.Random(42).shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
