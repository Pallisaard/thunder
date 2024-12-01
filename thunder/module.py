import abc
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, Generic, final
from pathlib import Path

import jax

from optax import GradientTransformation
from clu import metrics
import orbax.checkpoint as ocp

from thunder.utils import AutoInit
from thunder.types import State, ArrayType


class ThunderModule(AutoInit, Generic[State], abc.ABC):
    def __init__(self):
        super().__init__()
        self.metrics_history: Mapping[str, list[int | float]] = defaultdict()
        self.state = self.configure_state()

    @abc.abstractmethod
    def configure_state(self) -> State: ...

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

    def log(self, name: str, value: int | float) -> None:
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
        self, state: State, path: Path | str, prefix: str | None = None
    ) -> None:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        checkpoint_dir = str(path / (prefix if prefix else "checkpoint"))
        checkpointer.save(state, checkpoint_dir)

    @final
    @staticmethod
    def load_checkpoint(path: Path) -> State:
        # Create a Checkpointer object
        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

        # Load the checkpoint
        state = checkpointer.restore(str(path))

        return state
