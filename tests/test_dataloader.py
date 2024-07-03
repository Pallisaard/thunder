from thunder.dataset import Dataset
from thunder.dataloader import (
    create_dataloader,
    create_jax_dataloader,
    create_numpy_dataloader,
    create_list_dataloader,
    JaxDataLoader,
    NumpyDataLoader,
    ListDataLoader,
)

import numpy as np
import jax.numpy as jnp


class FloatDataset(Dataset):
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    def getitem(self, index: int):
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
    dl = create_dataloader(ds, batch_size=1, convert_to="list")
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]

    for i, v in enumerate(dl):
        assert v == [test_data[i]]  # Comes out as lists of things


def test_generic_dataloader_to_list_collate():
    def collate_fn(batch: list[float]) -> list[float]:
        return [elm * elm + 2.0 for elm in batch]

    ds = FloatDataset()
    dl = create_dataloader(ds, 1, "list", collate_fn)
    test_data = [3.0, 6.0, 11.0, 18.0, 27.0]

    for i, v in enumerate(dl):
        assert v == [test_data[i]]


def test_generic_dataloader_to_numpy_no_collate_with_float_data():
    ds = FloatDataset()
    dl = create_dataloader(ds, 2, "numpy")
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, v in enumerate(dl):
        assert type(test_data) == type(v)
        assert np.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_numpy_collate_with_float_data():
    def collate_fn(batch: list[float]) -> list[float]:
        return [x * x for x in batch]

    ds = FloatDataset()
    dl = create_dataloader(ds, 2, "numpy", collate_fn)
    test_data = np.array([1.0, 4.0, 9.0, 16.0, 25.0])

    for i, v in enumerate(dl):
        assert type(test_data) == type(v)
        assert np.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_numpy_no_collate_with_np_data():
    ds = NumpyDataset()
    dl = create_dataloader(ds, 2, "numpy")
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, v in enumerate(dl):
        assert type(test_data) == type(v)
        assert np.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_numpy_collate_with_np_data():
    def collate_fn(batch: list[np.ndarray]) -> list[np.ndarray]:
        return [np.pad(np.expand_dims(x, 0), 1) for x in batch]

    ds = NumpyDataset()
    dl = create_dataloader(ds, 2, "numpy", collate_fn)
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
        assert type(test_data) == type(v)
        assert np.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_jax_no_collate_with_float_data():
    ds = FloatDataset()
    dl = create_dataloader(ds, 2, "jax")
    test_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, v in enumerate(dl):
        assert type(test_data) == type(v)
        assert jnp.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_jax_collate_with_float_data():
    def collate_fn(batch: list[float]) -> list[float]:
        return [x * x for x in batch]

    ds = FloatDataset()
    dl = create_dataloader(ds, 2, "jax", collate_fn)
    test_data = jnp.array([1.0, 4.0, 9.0, 16.0, 25.0])

    for i, v in enumerate(dl):
        assert type(test_data) == type(v)
        assert jnp.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_jax_no_collate_with_jax_data():
    ds = JaxDataset()
    dl = create_dataloader(ds, 2, "jax")
    test_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, v in enumerate(dl):
        assert type(test_data) == type(v)
        assert jnp.all(test_data[2 * i : 2 * i + 2] == v)


def test_generic_dataloader_to_jax_collate_with_jax_data():
    def collate_fn(batch: list[jnp.ndarray]) -> list[jnp.ndarray]:
        return [jnp.pad(jnp.expand_dims(x, 0), 1) for x in batch]

    ds = JaxDataset()
    dl = create_dataloader(ds, 2, "jax", collate_fn)
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
        assert type(test_data) == type(v)
        assert jnp.all(test_data[2 * i : 2 * i + 2] == v)


def test_list_dataloader_no_collate_float_data():
    ds = FloatDataset()
    dl = create_list_dataloader(ds, 2)
    test_data = [[1.0, 2.0], [3.0, 4.0], [5.0]]

    for i, v in enumerate(dl):
        assert test_data[i] == v


def test_list_dataloader_collate_float_data():
    def collate_fn(batch: list[float]) -> list[float]:
        return [x * x for x in batch]

    ds = FloatDataset()
    dl = create_list_dataloader(ds, 2, collate_fn)
    test_data = [[1.0, 4.0], [9.0, 16.0], [25.0]]

    for i, v in enumerate(dl):
        assert test_data[i] == v


def test_numpy_dataloader_no_collate_float_data():
    ds = FloatDataset()
    dl = create_numpy_dataloader(ds, 2)
    test_data = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0])]

    for i, v in enumerate(dl):
        assert np.all(test_data[i] == v)


def test_numpy_dataloader_collate_float_data():
    def collate_fn(batch: list[float]) -> list[float]:
        return [x * x for x in batch]

    ds = FloatDataset()
    dl = create_numpy_dataloader(ds, 2, collate_fn)
    test_data = [np.array([1.0, 4.0]), np.array([9.0, 16.0]), np.array([25.0])]

    for i, v in enumerate(dl):
        assert np.all(test_data[i] == v)


def test_jax_dataloader_no_collate_float_data():
    ds = FloatDataset()
    dl = create_jax_dataloader(ds, 2)
    test_data = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0])]

    for i, v in enumerate(dl):
        assert jnp.all(test_data[i] == v)


def test_jax_dataloader_collate_float_data():
    def collate_fn(batch: list[float]) -> list[float]:
        return [x * x for x in batch]

    ds = FloatDataset()
    dl = create_jax_dataloader(ds, 2, collate_fn)
    test_data = [jnp.array([1.0, 4.0]), jnp.array([9.0, 16.0]), jnp.array([25.0])]

    for i, v in enumerate(dl):
        assert jnp.all(test_data[i] == v)


def test_jax_dataloader_from_arguments_no_collate():
    ds = FloatDataset()
    dl = JaxDataLoader.from_arguments(ds, 2)
    test_data = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0])]

    for i, v in enumerate(dl):
        assert jnp.all(test_data[i] == v)


def test_jax_dataloader_from_arguments_collate():
    def collate_fn(batch: list[float]) -> list[float]:
        return [x * x for x in batch]

    ds = FloatDataset()
    dl = JaxDataLoader.from_arguments(ds, 2, collate_fn)
    test_data = [jnp.array([1.0, 4.0]), jnp.array([9.0, 16.0]), jnp.array([25.0])]

    for i, v in enumerate(dl):
        assert jnp.all(test_data[i] == v)


def test_numpy_dataloader_from_arguments_no_collate():
    ds = FloatDataset()
    dl = NumpyDataLoader.from_arguments(ds, 2)
    test_data = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0])]

    for i, v in enumerate(dl):
        assert np.all(test_data[i] == v)


def test_numpy_dataloader_from_arguments_collate():
    def collate_fn(batch: list[float]) -> list[float]:
        return [x * x for x in batch]

    ds = FloatDataset()
    dl = NumpyDataLoader.from_arguments(ds, 2, collate_fn)
    test_data = [np.array([1.0, 4.0]), np.array([9.0, 16.0]), np.array([25.0])]

    for i, v in enumerate(dl):
        assert np.all(test_data[i] == v)


def test_list_dataloader_from_arguments_no_collate():
    ds = FloatDataset()
    dl = ListDataLoader.from_arguments(ds, 2)
    test_data = [[1.0, 2.0], [3.0, 4.0], [5.0]]

    for i, v in enumerate(dl):
        assert test_data[i] == v


def test_list_dataloader_from_arguments_collate():
    def collate_fn(batch: list[float]) -> list[float]:
        return [x * x for x in batch]

    ds = FloatDataset()
    dl = ListDataLoader.from_arguments(ds, 2, collate_fn)
    test_data = [[1.0, 4.0], [9.0, 16.0], [25.0]]

    for i, v in enumerate(dl):
        assert test_data[i] == v
