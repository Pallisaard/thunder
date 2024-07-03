from typing import (
    Callable,
    Iterator,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import jax.numpy as jnp

from thunder.dataset import Dataset
from thunder.types import NPArray, JaxArray

T_co = TypeVar("T_co", covariant=True)
U_co = TypeVar("U_co", covariant=True)
CONVERT_LITERALS = Literal["list", "numpy", "jax"]
CONVERT_TYPES = Union[List[T_co], NPArray, JaxArray]
COLLATE_FN = Callable[[List[T_co]], List[T_co]]

# Define a type variable for the convert_to argument
ConvertToLiteral = TypeVar("ConvertToLiteral", bound=CONVERT_LITERALS)


class BaseDataLoader[T_co]:
    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: int,
        collate_fn: Optional[COLLATE_FN[T_co]] = None,
        prefetch_factor: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.prefetch_factor = prefetch_factor
        self.prefetch_buffer = []

    def __iter__(self):
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch = [
                self.dataset[idx]
                for idx in range(
                    start_idx, min(start_idx + self.batch_size, len(self.dataset))
                )
            ]

            if self.collate_fn:
                batch = self.collate_fn(batch)

            yield batch


class NumpyDataLoader[T_co]:
    def __init__(
        self,
        base_dataloader: BaseDataLoader[T_co],
    ) -> None:
        self.base_dataloader = base_dataloader

    def __iter__(self) -> Iterator[NPArray]:
        for batch in self.base_dataloader:
            batch = convert_batch_to_numpy(batch)
            yield batch

    @staticmethod
    def from_arguments(
        dataset: Dataset[T_co],
        batch_size: int,
        collate_fn: Optional[COLLATE_FN[T_co]] = None,
    ):
        return NumpyDataLoader(BaseDataLoader(dataset, batch_size, collate_fn))


class JaxDataLoader[T_co]:
    def __init__(
        self,
        base_dataloader: BaseDataLoader[T_co],
    ) -> None:
        self.base_dataloader = base_dataloader

    def __iter__(self) -> Iterator[JaxArray]:
        for batch in self.base_dataloader:
            batch = convert_batch_to_jax(batch)
            yield batch

    @staticmethod
    def from_arguments(
        dataset: Dataset[T_co],
        batch_size: int,
        collate_fn: Optional[COLLATE_FN[T_co]] = None,
    ):
        return JaxDataLoader(BaseDataLoader(dataset, batch_size, collate_fn))


class ListDataLoader[T_co]:
    def __init__(
        self,
        base_dataloader: BaseDataLoader[T_co],
    ) -> None:
        self.base_dataloader = base_dataloader

    def __iter__(self) -> Iterator[List]:
        for batch in self.base_dataloader:
            yield batch

    @staticmethod
    def from_arguments(
        dataset: Dataset[T_co],
        batch_size: int,
        collate_fn: Optional[COLLATE_FN[T_co]] = None,
    ):
        return ListDataLoader(BaseDataLoader(dataset, batch_size, collate_fn))


def convert_batch_to_numpy[T_co](batch: List[T_co]) -> NPArray:
    return np.array(batch)


def convert_batch_to_jax[T_co](batch: List[T_co]) -> JaxArray:
    return jnp.array(batch)


def convert_batch_to_list[T_co](batch: List[T_co]) -> List[T_co]:
    return list(batch)


def create_dataloader[T_co](
    dataset: Dataset[T_co],
    batch_size: int,
    convert_to: CONVERT_LITERALS,
    collate_fn: Optional[COLLATE_FN[T_co]] = None,
):
    base_dataloader = BaseDataLoader(dataset, batch_size, collate_fn)
    if convert_to == "list":
        return cast(ListDataLoader, ListDataLoader(base_dataloader))
    elif convert_to == "numpy":
        return cast(NumpyDataLoader, NumpyDataLoader(base_dataloader))
    elif convert_to == "jax":
        return cast(JaxDataLoader, JaxDataLoader(base_dataloader))
    else:
        raise ValueError("'convert_to' must be set to either 'list', 'numpy, or 'jax'")


def create_jax_dataloader[T_co](
    dataset: Dataset[T_co],
    batch_size: int,
    collate_fn: Optional[COLLATE_FN[T_co]] = None,
):
    return JaxDataLoader.from_arguments(dataset, batch_size, collate_fn)


def create_numpy_dataloader[T_co](
    dataset: Dataset[T_co],
    batch_size: int,
    collate_fn: Optional[COLLATE_FN[T_co]] = None,
):
    return NumpyDataLoader.from_arguments(dataset, batch_size, collate_fn)


def create_list_dataloader[T_co](
    dataset: Dataset[T_co],
    batch_size: int,
    collate_fn: Optional[COLLATE_FN[T_co]] = None,
):
    return ListDataLoader.from_arguments(dataset, batch_size, collate_fn)
