from typing import Generic, Iterator, List, Optional, Type, TypeVar, Union
from thunder.types import CollateFn, DatasetLike
from thunder.data.sampler import BaseSampler, RandomSampler, SequentialSampler

T = TypeVar("T")


class DataIterator(Generic[T]):
    """Iterates over a dataset in batches, returning lists of items or collated batches.

    Supports any dataset that implements __getitem__ and __len__, including:
    - Thunder Dataset
    - HuggingFace Dataset
    - Custom implementations
    """

    def __init__(
        self,
        dataset: DatasetLike[T],
        batch_size: int,
        collate_fn: Optional[CollateFn[T]],
        shuffle: bool,
        replacement: bool,
        infinite: bool,
        sampler_cls: Optional[Type[BaseSampler]] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.replacement = replacement
        self.infinite = infinite
        self.sampler_cls = sampler_cls
        if not shuffle:
            sampler_cls = SequentialSampler
        elif sampler_cls is None:
            sampler_cls = RandomSampler
        self.sampler = sampler_cls(N=len(dataset), k=batch_size, replacement=replacement, infinite=infinite)

    def __iter__(self) -> Iterator[Union[List[T], T]]:
        return self

    def __next__(self) -> Union[List[T], T]:
        sampled_ids = next(self.sampler)
        batch = [self.dataset[idx] for idx in sampled_ids]

        if self.collate_fn is not None:
            batch = self.collate_fn(batch)

        return batch
