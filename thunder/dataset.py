from typing import Generic, Iterator, Optional
from thunder.utils import AutoInit
from thunder.types import AnyDataType

__all__ = ["Dataset", "IterableDataset"]


class Dataset(AutoInit, Generic[AnyDataType]):
    dtype: Optional[type] = None

    def getitem(self, index: int) -> AnyDataType:
        raise NotImplementedError("Subclasses must implement `getitem(self, index)`.")

    def len(self) -> int:
        raise NotImplementedError("Subclasses must implement `len(self)`.")

    def __getitem__(self, index: int) -> AnyDataType:
        return self.getitem(index)

    def __len__(self) -> int:
        return self.len()


class IterableDataset(AutoInit, Generic[AnyDataType]):
    def next(self) -> AnyDataType:
        raise NotImplementedError("Subclasses must implement `next(self)`.")

    def __iter__(self) -> Iterator[AnyDataType]:
        return self

    def __next__(self) -> AnyDataType:
        return self.next()
