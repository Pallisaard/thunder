from typing import Callable, List, Protocol, TypeVar

import numpy as np
import jax
from flax.training import train_state

# Type aliases for specific array types
NPArray = np.ndarray
JaxArray = jax.Array

# Generic type variable for dataset items (covariant for Protocol)
T_co = TypeVar("T_co", covariant=True)


# Protocol for dataset-like objects (works with both Thunder and HF datasets)
class DatasetLike(Protocol[T_co]):
    """Protocol that any dataset must satisfy to work with DataLoader.

    Uses structural typing (Protocol) to accept any object with __getitem__ and __len__.
    This includes:
    - Thunder Dataset
    - HuggingFace Dataset
    - PyTorch Dataset
    - TensorFlow Dataset
    - Any custom implementation with these methods

    No import of these libraries is needed - the Protocol uses duck typing!
    """

    def __getitem__(self, index: int) -> T_co:
        """Get item at index."""
        ...

    def __len__(self) -> int:
        """Get dataset length."""
        ...


# Generic type variable for non-protocol use
T = TypeVar("T")

# Collate function must preserve type structure
# Takes a list of items, returns a batched version
CollateFn = Callable[[List[T]], T]

# State type for training
State = TypeVar("State", bound=train_state.TrainState)
