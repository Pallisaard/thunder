from thunder.data.dataset import Dataset
from thunder.data.dataloader import DataLoader

import numpy as np
import jax.numpy as jnp


class FloatDataset(Dataset[float]):
    """Dataset that returns individual floats."""

    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    def getitem(self, index: int) -> float:
        return self.data[index]

    def len(self) -> int:
        return len(self.data)


class NumpyDataset(Dataset[np.ndarray]):
    """Dataset that returns NumPy arrays."""

    data = [
        np.array(1.0),
        np.array(2.0),
        np.array(3.0),
        np.array(4.0),
        np.array(5.0),
    ]

    def getitem(self, index: int) -> np.ndarray:
        return self.data[index]

    def len(self) -> int:
        return len(self.data)


class JaxDataset(Dataset[jnp.ndarray]):
    """Dataset that returns JAX arrays."""

    data = [
        jnp.array(1.0),
        jnp.array(2.0),
        jnp.array(3.0),
        jnp.array(4.0),
        jnp.array(5.0),
    ]

    def getitem(self, index: int) -> jnp.ndarray:
        return self.data[index]

    def len(self) -> int:
        return len(self.data)


def test_dataloader_with_float_data():
    """Test that floats are automatically converted to JAX arrays by default_collate."""
    ds = FloatDataset()
    dl = DataLoader(ds, batch_size=2)
    test_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, batch in enumerate(dl):
        assert isinstance(batch, jnp.ndarray)
        expected = test_data[2 * i : 2 * i + 2]
        assert jnp.allclose(batch, expected)


def test_dataloader_with_custom_collate():
    """Test custom collate function."""

    def collate_fn(batch: list[float]) -> jnp.ndarray:
        # Transform each element and return as JAX array
        transformed = [elm * elm + 2.0 for elm in batch]
        return jnp.array(transformed)

    ds = FloatDataset()
    # Type checker can't verify that primitives batch to arrays at runtime
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)  # type: ignore[arg-type]
    test_data = jnp.array([3.0, 6.0, 11.0, 18.0, 27.0])

    for i, batch in enumerate(dl):
        assert isinstance(batch, jnp.ndarray)
        expected = test_data[2 * i : 2 * i + 2]
        assert jnp.allclose(batch, expected)


def test_dataloader_with_numpy_data():
    """Test that NumPy arrays are converted to JAX arrays."""
    ds = NumpyDataset()
    dl = DataLoader(ds, batch_size=2)
    test_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, batch in enumerate(dl):
        assert isinstance(batch, jnp.ndarray)  # NumPy → JAX conversion
        expected = test_data[2 * i : 2 * i + 2]
        assert jnp.allclose(batch, expected)


def test_dataloader_with_numpy_custom_collate():
    """Test custom collate with NumPy arrays can still return NumPy if desired."""

    def collate_fn(batch: list[np.ndarray]) -> np.ndarray:
        # Custom collate that returns NumPy (user's choice)
        # Pad each array and stack
        return np.stack([np.pad(np.expand_dims(x, 0), 1) for x in batch])

    ds = NumpyDataset()
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)  # type: ignore[arg-type]
    test_data = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )

    for i, batch in enumerate(dl):
        assert isinstance(batch, np.ndarray)  # Custom collate can override default
        expected = test_data[2 * i : 2 * i + 2]
        assert np.allclose(batch, expected)


def test_dataloader_with_jax_data():
    """Test that JAX arrays are stacked properly."""
    ds = JaxDataset()
    dl = DataLoader(ds, batch_size=2)
    test_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for i, batch in enumerate(dl):
        assert isinstance(batch, jnp.ndarray)
        expected = test_data[2 * i : 2 * i + 2]
        assert jnp.allclose(batch, expected)


def test_dataloader_with_jax_custom_collate():
    """Test custom collate with JAX arrays."""

    def collate_fn(batch: list[jnp.ndarray]) -> jnp.ndarray:
        # Pad each array and stack
        return jnp.stack([jnp.pad(jnp.expand_dims(x, 0), 1) for x in batch])

    ds = JaxDataset()
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    test_data = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )

    for i, batch in enumerate(dl):
        assert isinstance(batch, jnp.ndarray)
        expected = test_data[2 * i : 2 * i + 2]
        assert jnp.allclose(batch, expected)


def test_dataloader_with_tuple_data():
    """Test that tuples are collated correctly."""

    class TupleDataset(Dataset[tuple[np.ndarray, int]]):
        def getitem(self, index: int) -> tuple[np.ndarray, int]:
            return np.array([index, index * 2]), index

        def len(self) -> int:
            return 5

    ds = TupleDataset()
    dl = DataLoader(ds, batch_size=2)

    for i, (features, labels) in enumerate(dl):
        assert isinstance(features, jnp.ndarray)  # NumPy → JAX conversion
        assert isinstance(labels, jnp.ndarray)  # Primitives → JAX arrays
        assert features.shape[0] == labels.shape[0]  # Same batch size


def test_dataloader_with_dict_data():
    """Test that dicts are collated correctly."""

    class DictDataset(Dataset[dict[str, np.ndarray]]):
        def getitem(self, index: int) -> dict[str, np.ndarray]:
            return {"input": np.array([index]), "target": np.array([index * 2])}

        def len(self) -> int:
            return 5

    ds = DictDataset()
    dl = DataLoader(ds, batch_size=2)

    for batch in dl:
        assert isinstance(batch, dict)
        assert "input" in batch and "target" in batch
        assert isinstance(batch["input"], jnp.ndarray)  # NumPy → JAX conversion
        assert isinstance(batch["target"], jnp.ndarray)  # NumPy → JAX conversion
        assert batch["input"].shape[0] == batch["target"].shape[0]  # Same batch size
