from dataclasses import dataclass
from typing import Dict, Tuple, TypeVar
import pathlib


import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
T_dict = Dict[str, T_co]
T_tuple = Tuple[T_co, ...]
T_stack = TypeVar("T_stack", T_tuple, T_dict)

NPArray = np.ndarray
JaxArray = jax.Array
Array = NPArray | JaxArray


# A dataclass named Epoch where the only value, an int, is an integer larger than -2
@dataclass
class Epoch:
    """Either a positive integer or -1 for infinite epochs."""

    value: int

    def __post_init__(self):
        assert (
            self.value > 0 or self.value == -1
        ), "Epoch value must be greater than 0 or equal to -1"


State = TypeVar("State", bound=train_state.TrainState)
Path = str | pathlib.Path


@dataclass
class PrefetchInfo[T_co]:
    buffer: list[T_co]
    size: int
    length: int
