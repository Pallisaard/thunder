from typing import Any, Generic
from thunder.module import ThunderModule
from collections.abc import Callable, Iterable

import jax

from thunder.types import State, Epoch, Array, Path


class Trainer(Generic[State]):
    def __init__(self, thunder_module: ThunderModule[State]):
        self.thunder_module = thunder_module
        self.state = self.thunder_module.state

    def train_epoch(self, state: State, train_dataloader: Iterable) -> State:
        state = self.thunder_module.on_train_start(state)

        train_step = jax.jit(self.thunder_module.train_step)
        for x, y in train_dataloader:
            state = train_step(state, x, y)

        state = self.thunder_module.on_train_epoch_end(state)
        return state

    def val_epoch(self, state: State, val_dataloader: Iterable) -> State:
        state = self.thunder_module.on_validation_start(state)

        val_step = jax.jit(self.thunder_module.val_step)
        for x, y in val_dataloader:
            state = val_step(state, x, y)

        state = self.thunder_module.on_validation_end(state)
        return state

    def fit(
        self,
        epochs: Epoch,
        train_dataloader: Iterable,
        val_dataloader: Iterable | None = None,
        test_dataloader: Iterable | None = None,
    ) -> None:
        state = self.state

        if epochs.value == -1:
            while True:
                self.train_epoch(state, train_dataloader)

                if val_dataloader is not None:
                    self.val_epoch(state, val_dataloader)
