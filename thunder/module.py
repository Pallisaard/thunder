import abc
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, Generic, final
import pathlib
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from optax import GradientTransformation
from clu import metrics
import orbax.checkpoint as ocp

from thunder.utils import AutoInitMeta
from thunder.types import State, Array, Path


class ThunderModule(Generic[State], abc.ABC, metaclass=AutoInitMeta):
    def __init__(self):
        super().__init__()
        self.metrics_history: Mapping[str, list[Array | int | float]] = defaultdict()
        self.state = self.configure_state()

    @abc.abstractmethod
    def configure_state(self) -> State: ...

    def make_loss_fn(self, state: State, x: Any, y: Any) -> Callable[[Any], Any]:
        """
        Returns a function that computes the loss. When this method is defined, the user has access to `self.make_grad_fn` and `self.make_value_and_grad_fn` which can be used to simplify construction of gradient functions. It is not implemented by default
        """
        ...

    def make_grad_fn(
        self, state: State, x: Any, y: Any, has_aux: bool = False, **kwargs
    ) -> Callable:
        """
        Returns a function that computes the gradient of the loss with respect to the parameters. Equivalent to `jax.grad(self.make_loss_fn(state, x, y), **kwargs)`.
        """
        try:
            loss_fn = self.make_loss_fn(state, x, y)
            grad_fn = jax.grad(loss_fn, has_aux=has_aux, **kwargs)
            return grad_fn
        except NotImplementedError:
            raise NotImplementedError(
                "You must implement `make_loss_fn` to use `make_grad_fn`"
            )

    def make_value_and_grad_fn(
        self, state: State, x: Any, y: Any, has_aux: bool = False, **kwargs
    ) -> Callable:
        """
        Returns a function that computes the loss and its gradient with respect to the parameters. Equivalent to `jax.value_and_grad(self.make_loss_fn(state, x, y), **kwargs)`.
        """
        try:
            loss_fn = self.make_loss_fn(state, x, y)
            value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=has_aux, **kwargs)
            return value_and_grad_fn
        except NotImplementedError:
            raise NotImplementedError(
                "You must implement `make_loss_fn` to use `make_value_and_grad_fn`"
            )

    @abc.abstractmethod
    def train_step(self, state: State, x: Any, y: Any) -> State: ...

    def validation_step(self, state: State, x: Any, y: Any) -> State:
        raise NotImplementedError(
            "You must implement `validation_step` to be able to validate the model"
        )

    def val_step(self, state: State, x: Any, y: Any) -> State:
        return state

    def test_step(self, state: State, x: Any, y: Any) -> State:
        return state

    def log(self, name: str, value: Array | int | float) -> None:
        self.metrics_history[name].append(value)

    @final
    def collect_metrics(self, state, **kwargs) -> State:
        new_metrics = state.metrics.single_from_model_output(**kwargs)
        state = state.replace(metrics=new_metrics)
        return state

    def on_train_start(self, state: State) -> State:
        return state

    def on_train_end(self, state: State) -> State:
        return state

    def on_train_epoch_start(self, state: State) -> State:
        return state

    def on_train_epoch_end(self, state: State) -> State:
        return state

    def on_validation_start(self, state: State) -> State:
        return state

    def on_validation_end(self, state: State) -> State:
        return state

    def on_validation_epoch_start(self, state: State) -> State:
        return state

    def on_validation_epoch_end(self, state: State) -> State:
        return state

    @final
    def save_checkpoint(
        self, state: State, path: Path, prefix: str | None = None
    ) -> None:
        path = pathlib.Path(path)
        prefix = "" if prefix is None else "_" + prefix
        path = path / "checkpoints" / (self.__class__.__name__ + prefix)
        path.mkdir(parents=True, exist_ok=True)

    @final
    @staticmethod
    def load_checkpoint(path: Path) -> State: ...
