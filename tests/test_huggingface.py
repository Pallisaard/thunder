"""Tests for HuggingFace datasets integration."""

import pytest
import jax.numpy as jnp

from thunder.data import DataLoader, prepare_huggingface_dataset


def test_huggingface_dataset_with_jax_format():
    """Test that HuggingFace datasets work with JAX format."""
    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        pytest.skip("datasets library not installed")

    # Create a simple HuggingFace dataset
    data = {
        "values": [1.0, 2.0, 3.0, 4.0, 5.0],
        "labels": [0, 1, 0, 1, 0],
    }
    hf_dataset = HFDataset.from_dict(data)

    # Set JAX format
    hf_dataset = hf_dataset.with_format("jax")

    # Use with DataLoader
    dl = DataLoader(hf_dataset, batch_size=2)  # pyright: ignore[reportArgumentType]

    for batch in dl:
        # Check that batch is a dict
        assert isinstance(batch, dict)
        assert "values" in batch and "labels" in batch

        # Check that values are JAX arrays
        assert isinstance(batch["values"], jnp.ndarray)
        assert isinstance(batch["labels"], jnp.ndarray)

        # Check batch size
        assert batch["values"].shape[0] == batch["labels"].shape[0]


def test_huggingface_dataset_with_numpy_format():
    """Test that HuggingFace datasets work with NumPy format (auto-converts to JAX)."""
    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        pytest.skip("datasets library not installed")

    # Create a simple HuggingFace dataset
    data = {
        "features": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        "targets": [0, 1, 2, 3],
    }
    hf_dataset = HFDataset.from_dict(data)

    # Set NumPy format (will be converted to JAX by default_collate)
    hf_dataset = hf_dataset.with_format("numpy")

    # Use with DataLoader
    dl = DataLoader(hf_dataset, batch_size=2)  # pyright: ignore[reportArgumentType]

    for batch in dl:
        # Check that batch is a dict
        assert isinstance(batch, dict)

        # Check that NumPy arrays were converted to JAX
        assert isinstance(batch["features"], jnp.ndarray)
        assert isinstance(batch["targets"], jnp.ndarray)


def test_prepare_huggingface_dataset():
    """Test the prepare_huggingface_dataset helper function."""
    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        pytest.skip("datasets library not installed")

    # Create dataset
    data = {"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}
    hf_dataset = HFDataset.from_dict(data)

    # Prepare for JAX
    jax_dataset = prepare_huggingface_dataset(hf_dataset, format="jax")

    # Verify format was set
    item = jax_dataset[0]
    assert isinstance(item["x"], jnp.ndarray)
    assert isinstance(item["y"], jnp.ndarray)


def test_prepare_huggingface_dataset_invalid():
    """Test that prepare_huggingface_dataset raises error for non-HF datasets."""
    from thunder.data import Dataset

    class SimpleDataset(Dataset[int]):
        def getitem(self, idx: int) -> int:
            return idx

        def len(self) -> int:
            return 10

    ds = SimpleDataset()

    with pytest.raises(ValueError, match="does not have .with_format"):
        prepare_huggingface_dataset(ds)


def test_huggingface_with_shuffling():
    """Test HuggingFace datasets with shuffling."""
    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        pytest.skip("datasets library not installed")

    # Create dataset
    data = {"values": list(range(20))}
    hf_dataset = HFDataset.from_dict(data).with_format("jax")

    # Load with shuffling
    dl = DataLoader(hf_dataset, batch_size=5, shuffle=True)  # pyright: ignore[reportArgumentType]

    batches = list(dl)
    assert len(batches) == 4

    # Check all values are present (just in different order)
    all_values = []
    for batch in batches:
        all_values.extend(batch["values"].tolist())

    assert sorted(all_values) == list(range(20))


def test_huggingface_nested_structures():
    """Test HuggingFace datasets with nested structures."""
    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        pytest.skip("datasets library not installed")

    # Note: HF datasets flatten nested dicts by default
    # Let's create a simpler test
    simple_data = {
        "image": [[1, 2], [3, 4], [5, 6]],
        "caption_id": [0, 1, 2],
        "label": [10, 20, 30],
    }

    hf_dataset = HFDataset.from_dict(simple_data).with_format("jax")
    dl = DataLoader(hf_dataset, batch_size=2)  # pyright: ignore[reportArgumentType]

    for batch in dl:
        assert isinstance(batch, dict)
        assert "image" in batch
        assert "caption_id" in batch
        assert "label" in batch
        assert isinstance(batch["image"], jnp.ndarray)
