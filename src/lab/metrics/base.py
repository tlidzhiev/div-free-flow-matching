from typing import Any, Self, cast

from clu import metrics as clu_metrics
from flax import struct


@struct.dataclass
class BaseMetric(clu_metrics.Metric):
    """
    Base class for metrics accumulated across batches and computed at evaluation time.

    Use 'configure' to create a metric subclass with static hyperparameters for jit.
    """

    @classmethod
    def configure(cls, **kwargs: Any) -> type[Self]:
        """
        Return a subclass of the metric with the given hyperparameters (class attributes).

        Raises
        ------
        TypeError
            If the metric has no such hyperparameter.
        """
        for name in kwargs:
            if not hasattr(cls, name):
                raise TypeError(f"{cls.__name__} has no hyperparameter '{name}'.")
        namespace = {'__module__': cls.__module__, '__qualname__': cls.__qualname__, **kwargs}
        return cast(type[Self], struct.dataclass(type(cls.__name__, (cls,), namespace)))
