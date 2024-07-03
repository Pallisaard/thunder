import abc
from typing import Generic, Iterator, Optional
from thunder.utils import AutoInitMeta
from thunder.types import T_co

__all__ = ["Dataset", "IterableDataset"]


class Dataset(abc.ABC, Generic[T_co], metaclass=AutoInitMeta):
    dtype: Optional[type] = None

    @abc.abstractmethod
    def getitem(self, index: int) -> T_co: ...

    @abc.abstractmethod
    def len(self) -> int: ...

    def __getitem__(self, index: int) -> T_co:
        return self.getitem(index)

    def __len__(self) -> int:
        return self.len()


class IterableDataset(abc.ABC, Generic[T_co], metaclass=AutoInitMeta):
    @abc.abstractmethod
    def iter(self) -> Iterator[T_co]: ...

    def __iter__(self) -> Iterator[T_co]:
        return self.iter()
