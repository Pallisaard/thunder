from typing import Any
import jax
from flax import struct
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from clu import metrics
from optax import (
    softmax_cross_entropy,
    adamw,
    cosine_decay_schedule,
)

from thunder import module
from thunder.types import Array


@struct.dataclass
class MyMetrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")  # type: ignore


class CustomState(train_state.TrainState):
    metrics: MyMetrics


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(2)(x)
        x = nn.relu(x)
        return x


key = jax.random.PRNGKey(0)
base_model = Model()
test_x = jnp.ones((1, 2))
test_labels = jnp.zeros((1, 2), dtype=jnp.int32).at[0, 1].set(1)
base_params = base_model.init(key, test_x)
base_params["params"]["Dense_0"]["kernel"] = jnp.array([[1.0, 1.0], [1.0, 1.0]])
base_logits = base_model.apply(base_params, test_x)
base_loss = softmax_cross_entropy(base_logits, test_labels).mean()  # type: ignore

base_state = train_state.TrainState.create(
    apply_fn=base_model.apply, params=base_params, tx=adamw(1e-3)
)


def make_base_loss_fn(state, x, y):
    def base_loss_fn(params):
        logits = base_model.apply(params, x)
        return softmax_cross_entropy(logits, y).mean(), logits  # type: ignore

    return base_loss_fn


base_loss_fn = make_base_loss_fn(base_state, test_x, test_labels)
base_grad_fn = jax.grad(base_loss_fn, has_aux=True)
(base_grad2, base_logits2) = base_grad_fn(base_state.params)
base_value_and_grad_fn = jax.value_and_grad(base_loss_fn, has_aux=True)
(base_loss3, base_logits3), base_grad3 = base_value_and_grad_fn(base_state.params)

base_state = CustomState.create(
    apply_fn=base_model.apply,
    params=base_params,
    tx=adamw(cosine_decay_schedule(1e-3, 1000, 1e-5)),
    metrics=MyMetrics.empty(),
)

y = base_state.apply_fn(base_state.params, test_x)


class MyModule(module.ThunderModule):
    model = Model()

    def configure_state(self):
        return CustomState.create(
            apply_fn=self.model.apply,
            params=base_params,
            tx=adamw(1e-3),
            metrics=MyMetrics.empty(),
        )

    def make_loss_fn(self, state: CustomState, x, y):
        def loss_fn(params):
            logits = state.apply_fn(params, x)
            loss = softmax_cross_entropy(logits, y).mean()
            return loss, logits

        return loss_fn

    def train_step(self, state: CustomState, x, y):
        value_and_grad_fn = self.make_value_and_grad_fn(state, x, y, has_aux=True)
        (loss, logits), grads = value_and_grad_fn(state.params)
        state = state.apply_gradients(grads=grads)

        state = self.collect_metrics(state, logits=logits, labels=y, loss=loss)
        self.log("step_loss", state.metrics.loss.compute())
        return state

    def on_train_epoch_end(self, state: CustomState) -> CustomState:
        self.log("epoch_loss", state.metrics.loss.compute())
        self.log("epoch_accuracy", state.metrics.accuracy.compute())
        state = state.replace(metrics=MyMetrics.empty())
        return state

    def validation_step(self, state: CustomState, x, y):
        loss_fn = self.make_loss_fn(state, x, y)
        loss, logits = loss_fn(state.params)
        state = self.collect_metrics(state, logits=logits, labels=y, loss=loss)
        return state

    def on_validation_epoch_end(self, state: CustomState) -> CustomState:
        self.log("val_loss", state.metrics.loss.compute())
        self.log("val_accuracy", state.metrics.accuracy.compute())
        state = state.replace(metrics=MyMetrics.empty())
        return state


def test_init():
    my_module = MyModule()
    assert my_module.model == Model()
    assert my_module.state is not None
    assert my_module.state.apply_fn is not None
    assert my_module.state.params == base_params
    assert my_module.state.tx is not None
    assert my_module.state.metrics is not None


def test_make_loss_fn():
    my_module = MyModule()
    loss_fn = my_module.make_loss_fn(my_module.state, test_x, test_labels)
    loss, logits = loss_fn(my_module.state.params)
    assert jnp.all(loss == base_loss)
    assert jnp.all(logits == base_logits)


def test_make_grad_fn():
    my_module = MyModule()
    grad_fn = my_module.make_grad_fn(my_module.state, test_x, test_labels, has_aux=True)
    grad, logits = grad_fn(my_module.state.params)
    assert jnp.all(logits == base_logits2)
    assert isinstance(grad, dict)
    assert isinstance(base_grad2, dict)
    grad_eqs = jax.tree.map(lambda x, y: jnp.all(x == y), grad, base_grad2)
    assert jax.tree.all(grad_eqs)


def test_make_value_and_grad_fn():
    my_module = MyModule()
    value_and_grad_fn = my_module.make_value_and_grad_fn(
        my_module.state, test_x, test_labels, has_aux=True
    )
    (loss, logits), grad = value_and_grad_fn(my_module.state.params)
    assert jnp.all(loss == base_loss3)
    assert jnp.all(logits == base_logits2)
    grad_eqs = jax.tree.map(lambda x, y: jnp.all(x == y), grad, base_grad3)
    assert jax.tree.all(grad_eqs)
