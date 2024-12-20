from thunder.dataset import Dataset
from thunder.dataloader import DataLoader

import numpy as np
import jax.numpy as jnp


class FloatDataset(Dataset[float]):
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    def getitem(self, index: int) -> float:
        return self.data[index]

    def len(self) -> int:
        return len(self.data)


class NumpyDataset(Dataset):
    data = [
        np.array(1.0),
        np.array(2.0),
        np.array(3.0),
        np.array(4.0),
        np.array(5.0),
    ]

    def getitem(self, index: int):
        return self.data[index]

    def len(self) -> int:
        return len(self.data)


class JaxDataset(Dataset):
    data = [
        jnp.array(1.0),
        jnp.array(2.0),
        jnp.array(3.0),
        jnp.array(4.0),
        jnp.array(5.0),
    ]

    def getitem(self, index: int):
        return self.data[index]

    def len(self) -> int:
        return len(self.data)


def test_generic_dataloader_to_list_no_collate():
    ds = FloatDataset()
    dl = DataLoader(ds, batch_size=1, convert_to="list")
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]

    for i, v in enumerate(dl):
        assert v == [test_data[i]]  # Comes out as lists of things


def test_generic_dataloader_to_list_collate():
    def collate_fn(batch: list[float]) -> list[float]:
        return [elm * elm + 2.0 for elm in batch]

    ds = FloatDataset()
    dl = DataLoader(ds, batch_size=1, convert_to="list", collate_fn=collate_fn)
    test_data = [3.0, 6.0, 11.0, 18.0, 27.0]

    for i, v in enumerate(dl):
        assert v == [test_data[i]]


def test_generic_dataloader_to_numpy_no_collate_with_float_data():
    ds = FloatDataset()
    dl = DataLoader(ds, batch_size=2, convert_to="numpy")
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, v in enumerate(dl):
        assert isinstance(v, type(test_data))
        assert np.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_numpy_collate_with_float_data():
    def collate_fn(batch: list[float]) -> list[float]:
        return [x * x for x in batch]

    ds = FloatDataset()
    dl = DataLoader(ds, batch_size=2, convert_to="numpy", collate_fn=collate_fn)
    test_data = np.array([1.0, 4.0, 9.0, 16.0, 25.0])

    for i, v in enumerate(dl):
        assert isinstance(v, type(test_data))
        assert np.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_numpy_no_collate_with_np_data():
    ds = NumpyDataset()
    dl = DataLoader(ds, batch_size=2, convert_to="numpy")
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, v in enumerate(dl):
        assert isinstance(v, type(test_data))
        assert np.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_numpy_collate_with_np_data():
    def collate_fn(batch: list[np.ndarray]) -> list[np.ndarray]:
        return [np.pad(np.expand_dims(x, 0), 1) for x in batch]

    ds = NumpyDataset()
    dl = DataLoader(ds, batch_size=2, convert_to="numpy", collate_fn=collate_fn)
    test_data = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )

    for i, v in enumerate(dl):
        assert isinstance(v, type(test_data))
        assert np.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_jax_no_collate_with_float_data():
    ds = FloatDataset()
    dl = DataLoader(ds, batch_size=2, convert_to="jax")
    test_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, v in enumerate(dl):
        assert isinstance(v, type(test_data))
        assert jnp.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_jax_collate_with_float_data():
    def collate_fn(batch: list[float]) -> list[float]:
        return [x * x for x in batch]

    ds = FloatDataset()
    dl = DataLoader(ds, batch_size=2, convert_to="jax", collate_fn=collate_fn)
    test_data = jnp.array([1.0, 4.0, 9.0, 16.0, 25.0])

    for i, v in enumerate(dl):
        assert isinstance(v, type(test_data))
        assert jnp.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_jax_no_collate_with_jax_data():
    ds = JaxDataset()
    dl = DataLoader(ds, batch_size=2, convert_to="jax")
    test_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, v in enumerate(dl):
        assert isinstance(v, type(test_data))
        assert jnp.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_jax_collate_with_jax_data():
    def collate_fn(batch: list[jnp.ndarray]) -> list[jnp.ndarray]:
        return [jnp.pad(jnp.expand_dims(x, 0), 1) for x in batch]

    ds = JaxDataset()
    dl = DataLoader(ds, batch_size=2, convert_to="jax", collate_fn=collate_fn)
    test_data = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )

    for i, v in enumerate(dl):
        assert isinstance(v, type(test_data))
        assert jnp.all(test_data[2 * i : 2 * i + 2] == v)
