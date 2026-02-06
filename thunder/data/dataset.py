from typing import Generic, Iterator, Optional, TypeVar
from thunder.utils import AutoInit

__all__ = ["Dataset", "IterableDataset"]

T = TypeVar("T")


class Dataset(AutoInit, Generic[T]):
    """Base class for datasets. Returns items of type T."""

    dtype: Optional[type] = None

    def getitem(self, index: int) -> T:
        raise NotImplementedError("Subclasses must implement `getitem(self, index)`.")

    def len(self) -> int:
        raise NotImplementedError("Subclasses must implement `len(self)`.")

    def __getitem__(self, index: int) -> T:
        return self.getitem(index)

    def __len__(self) -> int:
        return self.len()


class IterableDataset(AutoInit, Generic[T]):
    """Base class for iterable datasets. Returns items of type T."""

    def next(self) -> T:
        raise NotImplementedError("Subclasses must implement `next(self)`.")

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        return self.next()
