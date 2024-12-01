from typing import Callable, List, Literal, TypeVar, Union
import pathlib


import numpy as np
import jax
from flax.training import train_state

AnyDataType = TypeVar("AnyDataType", covariant=True)

NPArray = np.ndarray
JaxArray = jax.Array
ElmArray = List[AnyDataType]
ArrayTypeUnion = Union[NPArray, JaxArray, ElmArray]

ArrayType = TypeVar("ArrayType", bound=ArrayTypeUnion)

ArrayTypeLiteral = Literal["list", "numpy", "jax"]

CollateFn = Callable[[ElmArray], ElmArray]


State = TypeVar("State", bound=train_state.TrainState)
