from typing import (
    Any,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
)
import numpy as np
import jax
import jax.numpy as jnp

from thunder.types import CollateFn, DatasetLike
from thunder.data.data_iterator import DataIterator
from thunder.data.prefetcher import Prefetcher

T = TypeVar("T")


def prepare_huggingface_dataset(dataset: Any, format: str = "jax") -> Any:
    """Prepare a HuggingFace dataset for use with Thunder DataLoader.

    This helper function ensures HuggingFace datasets are in the optimal format
    for JAX training. It calls .with_format() to convert arrays to the specified format.

    Args:
        dataset: HuggingFace Dataset object
        format: Output format for arrays. Options:
            - "jax": JAX arrays (recommended, default)
            - "numpy": NumPy arrays (will be converted to JAX by default_collate)
            - "torch": PyTorch tensors (not recommended)

    Returns:
        Dataset with specified format

    Example:
        >>> from datasets import load_dataset
        >>> from thunder.data import DataLoader, prepare_huggingface_dataset
        >>>
        >>> # Load HF dataset
        >>> hf_dataset = load_dataset("mnist", split="train")
        >>>
        >>> # Prepare for JAX (recommended)
        >>> jax_dataset = prepare_huggingface_dataset(hf_dataset, format="jax")
        >>>
        >>> # Use with DataLoader
        >>> dataloader = DataLoader(jax_dataset, batch_size=32)
        >>> for batch in dataloader:
        ...     # batch contains JAX arrays
        ...     train_step(batch)

    Note:
        You can also call .with_format("jax") directly on HF datasets:
        >>> hf_dataset = hf_dataset.with_format("jax")
    """
    if not hasattr(dataset, "with_format"):
        raise ValueError(
            "Dataset does not have .with_format() method. "
            "This helper is for HuggingFace datasets. "
            "For custom datasets, use Thunder Dataset class."
        )

    return dataset.with_format(format)


def default_collate(batch: List[Any]) -> Any:
    """
    Default collate function that intelligently batches data.

    This is a JAX-first collate function that converts ALL inputs to JAX arrays.
    This is the enforcement point: everything becomes JAX after collation.

    Design philosophy:
    - Dataset preprocessing uses NumPy (CPU-efficient, no GPU waste)
    - Collate converts everything to JAX (zero-copy for NumPy on CPU)
    - DataLoader yields JAX arrays (ready for @jax.jit training)

    Handles:
    - Tuples: Collates each element recursively
    - Dicts: Collates values for each key
    - NumPy arrays: Converts to JAX array (zero-copy on CPU)
    - JAX arrays: Stacks to JAX array
    - Lists: Converts to JAX array
    - NumPy scalars: Converts to JAX array (np.int64, np.float32, etc.)
    - Primitives (int, float, bool, str): Converts to JAX array

    Args:
        batch: List of items to collate

    Returns:
        Batched version of the input as JAX arrays, preserving structure

    Raises:
        ValueError: If items have inconsistent shapes or types
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate empty batch")

    elem = batch[0]

    # Handle tuples - recursively collate each element
    if isinstance(elem, tuple):
        return tuple(default_collate([item[i] for item in batch]) for i in range(len(elem)))

    # Handle dicts - recursively collate values for each key
    if isinstance(elem, dict):
        if not all(isinstance(item, dict) for item in batch):
            raise ValueError("All items in batch must be dicts if first item is a dict")

        keys = elem.keys()
        # Check all dicts have same keys
        for item in batch[1:]:
            if set(item.keys()) != set(keys):
                raise ValueError(
                    f"All dicts in batch must have same keys. Expected {set(keys)}, got {set(item.keys())}"
                )

        return {key: default_collate([item[key] for item in batch]) for key in keys}

    # Handle NumPy arrays - convert to JAX (zero-copy on CPU!)
    if isinstance(elem, np.ndarray):
        try:
            stacked = jnp.stack(batch, axis=0)
            return stacked
        except ValueError as e:
            shapes = [item.shape for item in batch]
            raise ValueError(f"Cannot stack arrays with inconsistent shapes: {shapes}") from e

    # Handle JAX arrays - stack to JAX
    if isinstance(elem, jax.Array):
        try:
            stacked = jnp.stack(batch, axis=0)
            return stacked
        except ValueError as e:
            shapes = [item.shape for item in batch]
            raise ValueError(f"Cannot stack JAX arrays with inconsistent shapes: {shapes}") from e

    # Handle lists - convert to JAX array
    if isinstance(elem, list):
        try:
            return jnp.array(batch)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert list batch to array: {e}") from e

    # Handle NumPy scalar types (np.int64, np.float32, etc.)
    if isinstance(elem, np.generic):
        return jnp.array(batch)

    # Handle primitives - convert to JAX array
    if isinstance(elem, (int, float, bool, str)):
        return jnp.array(batch)

    # Fallback - return as list if we don't know how to collate
    raise TypeError(f"default_collate: unsupported type {type(elem).__name__}")


def _setup_prefetching(
    data_iter: DataIterator[T],
    num_workers: int,
    prefetch_factor: Optional[int],
) -> Prefetcher[T]:
    if prefetch_factor is None:
        prefetch_factor = 2
    elif prefetch_factor < num_workers:
        raise ValueError(
            "prefetch_factor must be larger than or equal to the number of workers (prefetch_factor >= num_workers)."
        )
    return Prefetcher(data_iter, prefetch_factor, num_workers=num_workers)


class DataLoader(Generic[T]):
    """DataLoader for batching datasets with type preservation.

    Supports any dataset that implements __getitem__ and __len__, including:
    - Thunder Dataset
    - HuggingFace Dataset (use with_format("jax") for best performance)
    - Custom implementations

    Note: DataLoader always uses a collate function (default_collate if none provided),
    so it always yields batched items of type T, never List[T].

    Example with HuggingFace Dataset:
        >>> from datasets import load_dataset
        >>> hf_dataset = load_dataset("mnist", split="train")
        >>> hf_dataset = hf_dataset.with_format("jax")  # Recommended!
        >>> dataloader = DataLoader(hf_dataset, batch_size=32)
        >>> for batch in dataloader:
        ...     # batch is JAX arrays
        ...     pass
    """

    def __init__(
        self,
        dataset: DatasetLike[T],
        batch_size: int,
        shuffle: bool = False,
        replacement: bool = False,
        infinite: bool = False,
        collate_fn: CollateFn[T] = default_collate,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replacement = replacement
        self.infinite = infinite
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # Use default_collate if no custom collate function provided
        if collate_fn is None:
            collate_fn = default_collate
        self.collate_fn = collate_fn

        self.data_iterator = DataIterator(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            replacement=self.replacement,
            infinite=self.infinite,
        )

        if num_workers > 0:
            self.prefetcher = _setup_prefetching(self.data_iterator, self.num_workers, self.prefetch_factor)
        else:
            self.prefetcher = None

    def __iter__(self) -> Iterator[Any]:
        # Note: We return Iterator[Any] instead of Iterator[T] because
        # the actual return type depends on the collate function's behavior
        # at runtime. Type checkers can't verify this statically.
        if self.prefetcher is not None:
            for elm in self.prefetcher:
                yield elm
        else:
            for elm in self.data_iterator:
                yield elm
