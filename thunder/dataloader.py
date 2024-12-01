from typing import (
    Generic,
    Iterator,
    List,
    Optional,
    Union,
    overload,
)
import numpy as np
import jax.numpy as jnp

from thunder.dataset import Dataset
from thunder.types import (
    NPArray,
    JaxArray,
    AnyDataType,
    ArrayTypeLiteral,
    CollateFn,
)
from thunder.data_iterator import DataIterator
from thunder.prefetcher import Prefetcher


def _convert_list_batch_to_numpy(batch: List[AnyDataType]) -> NPArray:
    return np.asarray(batch)


def _convert_list_batch_to_jax(batch: List[AnyDataType]) -> JaxArray:
    return jnp.asarray(batch)


def _setup_prefetching(
    data_iter: DataIterator[AnyDataType],
    num_workers: int,
    prefetch_factor: Optional[int],
) -> Prefetcher[AnyDataType]:
    if prefetch_factor is None:
        prefetch_factor = 2
    elif prefetch_factor < num_workers:
        raise ValueError(
            "prefetch_factor must be larger than or equal to the number of workers (prefetch_factor >= num_workers)."
        )
    return Prefetcher(data_iter, prefetch_factor, num_workers=num_workers)


class DataLoader(Generic[AnyDataType]):
    def __init__(
        self,
        dataset: Dataset[AnyDataType],
        convert_to: ArrayTypeLiteral,
        batch_size: int,
        shuffle: bool = False,
        replacement: bool = False,
        infinite: bool = False,
        collate_fn: Optional[CollateFn] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.convert_to: ArrayTypeLiteral = convert_to
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replacement = replacement
        self.infinite = infinite

        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.data_iterator = DataIterator(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            replacement=self.replacement,
            infinite=self.infinite,
        )

        if num_workers > 0:
            self.prefetcher = _setup_prefetching(
                self.data_iterator, self.num_workers, self.prefetch_factor
            )
        else:
            self.prefetcher = None

    @overload
    def __iter__(self) -> Iterator[List[AnyDataType]]: ...  # type: ignore

    @overload
    def __iter__(self) -> Iterator[NPArray]: ...  # type: ignore

    @overload
    def __iter__(self) -> Iterator[JaxArray]: ...  # type: ignore

    def __iter__(self) -> Iterator[Union[NPArray, JaxArray, List[AnyDataType]]]:
        if self.prefetcher is not None:
            for elm in self.prefetcher:
                yield self._convert_batch(elm, convert_to=self.convert_to)
        else:
            for elm in self.data_iterator:
                yield self._convert_batch(elm, convert_to=self.convert_to)

    def _convert_batch(
        self, batch: List[AnyDataType], convert_to: ArrayTypeLiteral
    ) -> Union[NPArray, JaxArray, List[AnyDataType]]:
        if convert_to == "jax":
            return _convert_list_batch_to_jax(batch)
        elif convert_to == "numpy":
            return _convert_list_batch_to_numpy(batch)
        elif convert_to == "list":
            return batch
        else:
            raise ValueError(f"Unsupported convert_to value: {convert_to}")
