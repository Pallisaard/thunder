from thunder.dataset import Dataset, IterableDataset


def test_make_standard_dataset():
    class MyDataset(Dataset[float]):
        data = [1.0, 2.0, 3.0]

        def getitem(self, index: int) -> float:
            return self.data[index]

        def len(self) -> int:
            return len(self.data)

    a = MyDataset()
    assert a[2] == 3.0
    assert isinstance(a[2], float)
    assert len(a) == 3.0


def test_make_iterable_dataset():
    class MyIterableDataset(IterableDataset[float]):
        data = [1.0, 2.0, 3.0, 4.0]
        i = 0

        def next(self) -> float:
            if self.i >= len(self.data):
                raise StopIteration
            example = self.data[self.i]
            self.i += 1
            return example

    a = MyIterableDataset()
    for i, v in enumerate(a):
        assert i + 1.0 == v
