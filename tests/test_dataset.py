from typing import Iterator
from thunder.dataset import Dataset, IterableDataset


def test_make_standard_dataset():
    class MyDataset(Dataset[float]):
        data = [1.0, 2.0, 3.0]

        def getitem(self, index: int) -> float:
            return self.data[index]

        def len(self) -> int:
            return len(self.data)

    a = MyDataset()
    assert a[2] == 3
    assert len(a) == 3


def test_make_iterable_dataset():
    class MyIterableDataset(IterableDataset[float]):
        data = [1.0, 2.0, 3.0, 4.0]

        def iter(self) -> Iterator[float]:
            i = 0
            while i < len(self.data):
                yield self.data[i]
                i += 1

    a = MyIterableDataset()
    for i, v in enumerate(a):
        assert float(i + 1) == v
