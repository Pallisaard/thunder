from typing import Generic, Iterator, List, Optional
from thunder.dataset import Dataset
from thunder.types import AnyDataType, CollateFn
from thunder.sampler import BaseSampler, RandomSampler, SequentialSampler
from typing import Type


class DataIterator(Generic[AnyDataType]):
    def __init__(
        self,
        dataset: Dataset[AnyDataType],
        batch_size: int,
        collate_fn: Optional[CollateFn],
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
        self.sampler = sampler_cls(
            N=len(dataset), k=batch_size, replacement=replacement, infinite=infinite
        )

    def __iter__(self):
        return self

    def __next__(self) -> List[AnyDataType]:
        sampled_ids = next(self.sampler)
        batch = [self.dataset[idx] for idx in sampled_ids]

        if self.collate_fn is not None:
            batch = self.collate_fn(batch)

        return batch
