from thunder.data.dataset import Dataset
from thunder.data.data_iterator import DataIterator


class FloatDataset(Dataset[float]):
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    def getitem(self, index: int) -> float:
        return self.data[index]

    def len(self) -> int:
        return len(self.data)


def collate_fn(batch: list[float]) -> list[float]:
    return [x * 2 for x in batch]


def test_data_iterator_no_collate():
    ds = FloatDataset()
    iterator = DataIterator(
        ds,
        batch_size=2,
        collate_fn=None,
        shuffle=False,
        replacement=False,
        infinite=False,
    )
    expected_batches = [[1.0, 2.0], [3.0, 4.0], [5.0]]
    for batch, expected in zip(iterator, expected_batches):
        assert batch == expected


def test_data_iterator_with_collate():
    ds = FloatDataset()
    iterator = DataIterator(
        ds,
        batch_size=2,
        collate_fn=collate_fn,
        shuffle=False,
        replacement=False,
        infinite=False,
    )
    expected_batches = [[2.0, 4.0], [6.0, 8.0], [10.0]]
    for batch, expected in zip(iterator, expected_batches):
        assert batch == expected


def test_data_iterator_shuffle():
    ds = FloatDataset()
    iterator = DataIterator(
        ds,
        batch_size=2,
        collate_fn=None,
        shuffle=True,
        replacement=False,
        infinite=False,
    )
    batches = list(iterator)
    assert len(batches) == 3
    assert sorted(sum(batches, [])) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_data_iterator_replacement():
    ds = FloatDataset()
    iterator = DataIterator(
        ds,
        batch_size=2,
        collate_fn=None,
        shuffle=True,
        replacement=True,
        infinite=False,
    )
    batches = list(iterator)
    assert len(batches) == 3
    assert all(len(batch) == 2 for batch in batches[:2])
    assert len(batches[-1]) == 1


def test_data_iterator_infinite():
    ds = FloatDataset()
    iterator = DataIterator(
        ds,
        batch_size=2,
        collate_fn=None,
        shuffle=False,
        replacement=False,
        infinite=True,
    )
    batches = [next(iterator) for _ in range(4)]
    assert len(batches) == 4
    assert all(len(batch) == 2 for batch in batches)
