"""Tests for default_collate edge cases and error handling."""

import pytest
import numpy as np
import jax.numpy as jnp

from thunder.data import Dataset, DataLoader
from thunder.data.dataloader import default_collate


def test_collate_empty_batch():
    """Test that empty batch raises ValueError."""
    with pytest.raises(ValueError, match="Cannot collate empty batch"):
        default_collate([])


def test_collate_mixed_dict_types():
    """Test that mixing dicts and non-dicts raises ValueError."""
    batch = [{"a": 1}, 2, {"b": 3}]  # Mixed types
    with pytest.raises(ValueError, match="All items in batch must be dicts"):
        default_collate(batch)


def test_collate_mismatched_dict_keys():
    """Test that dicts with different keys raise ValueError."""
    batch = [
        {"x": 1, "y": 2},
        {"x": 3, "z": 4},  # Different key 'z' instead of 'y'
    ]
    with pytest.raises(ValueError, match="All dicts in batch must have same keys"):
        default_collate(batch)


def test_collate_inconsistent_numpy_shapes():
    """Test that NumPy arrays with inconsistent shapes raise ValueError."""
    batch = [
        np.array([1, 2, 3]),
        np.array([4, 5]),  # Different shape!
    ]
    with pytest.raises(ValueError, match="Cannot stack arrays with inconsistent shapes"):
        default_collate(batch)


def test_collate_inconsistent_jax_shapes():
    """Test that JAX arrays with inconsistent shapes raise ValueError."""
    batch = [
        jnp.array([[1, 2], [3, 4]]),  # Shape (2, 2)
        jnp.array([5, 6]),  # Shape (2,) - different!
    ]
    with pytest.raises(ValueError, match="Cannot stack JAX arrays with inconsistent shapes"):
        default_collate(batch)


def test_collate_unsupported_type():
    """Test that unsupported types raise TypeError."""

    class CustomObject:
        pass

    batch = [CustomObject(), CustomObject()]
    with pytest.raises(TypeError, match="default_collate: unsupported type"):
        default_collate(batch)


def test_collate_string_type():
    """Test that strings raise TypeError (JAX doesn't support string arrays)."""
    batch = ["hello", "world", "test"]

    # JAX doesn't support string arrays, so this should raise TypeError
    with pytest.raises(TypeError, match="not a valid JAX array type"):
        default_collate(batch)


def test_collate_bool_type():
    """Test that bools are handled correctly."""
    batch = [True, False, True, True]
    result = default_collate(batch)

    assert isinstance(result, jnp.ndarray)
    assert result.shape == (4,)


def test_collate_numpy_scalars():
    """Test that NumPy scalar types are handled."""
    batch = [np.int32(1), np.int32(2), np.int32(3)]
    result = default_collate(batch)

    assert isinstance(result, jnp.ndarray)
    assert result.shape == (3,)


def test_collate_nested_lists():
    """Test that nested lists work."""
    batch = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = default_collate(batch)

    assert isinstance(result, jnp.ndarray)
    assert result.shape == (2, 2, 2)


def test_dataloader_with_prefetching():
    """Test DataLoader with num_workers > 0 (prefetching)."""

    class SimpleDataset(Dataset[int]):
        def getitem(self, idx: int) -> int:
            return idx

        def len(self) -> int:
            return 10

    ds = SimpleDataset()

    # Test with prefetching
    dl = DataLoader(ds, batch_size=2, num_workers=2, prefetch_factor=4)

    batches = list(dl)
    assert len(batches) == 5

    # Verify all values are present
    all_values = []
    for batch in batches:
        all_values.extend(batch.tolist())
    assert sorted(all_values) == list(range(10))


def test_dataloader_prefetch_factor_validation():
    """Test that prefetch_factor < num_workers raises error during initialization."""

    class SimpleDataset(Dataset[int]):
        def getitem(self, idx: int) -> int:
            return idx

        def len(self) -> int:
            return 10

    ds = SimpleDataset()

    # This should raise an error during __init__
    with pytest.raises(ValueError, match="prefetch_factor must be larger than or equal"):
        dl = DataLoader(ds, batch_size=2, num_workers=4, prefetch_factor=2)


def test_dataset_base_class_not_implemented():
    """Test that base Dataset raises NotImplementedError."""
    from thunder.data import Dataset

    ds = Dataset()

    with pytest.raises(NotImplementedError, match="Subclasses must implement `getitem"):
        ds.getitem(0)

    with pytest.raises(NotImplementedError, match="Subclasses must implement `len"):
        ds.len()


def test_iterable_dataset_base_class_not_implemented():
    """Test that base IterableDataset raises NotImplementedError."""
    from thunder.data import IterableDataset

    ds = IterableDataset()

    with pytest.raises(NotImplementedError, match="Subclasses must implement `next"):
        ds.next()


def test_collate_with_mixed_numpy_types():
    """Test collating different NumPy dtypes."""
    batch = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
    ]
    result = default_collate(batch)

    assert isinstance(result, jnp.ndarray)
    assert result.shape == (2, 2)


def test_collate_nested_tuples():
    """Test collating nested tuples."""
    batch = [
        (np.array([1, 2]), (3, 4)),
        (np.array([5, 6]), (7, 8)),
    ]
    result = default_collate(batch)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], jnp.ndarray)  # Outer tuple's first element
    assert isinstance(result[1], tuple)  # Outer tuple's second element (nested tuple)
    assert len(result[1]) == 2
    assert isinstance(result[1][0], jnp.ndarray)  # Inner tuple converted to arrays
    assert isinstance(result[1][1], jnp.ndarray)
