"""Generic Flax NNX MLP building blocks.

This module holds the shared dense network used across the emulator package: a
stack of linear layers, a configurable hidden activation, and a final scalar
readout. Using ``flax.nnx`` keeps the code object-oriented while still fitting
cleanly into JAX and Optax training loops.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx


ActivationName = Literal["relu", "tanh", "gelu"]


def _activation_fn(name: ActivationName):
    """Translate a human-readable activation name into the JAX callable used at runtime."""
    activations = {
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "gelu": jax.nn.gelu,
    }
    return activations[name]


class DenseMLP(nnx.Module):
    """Simple dense MLP implemented with Flax NNX.

    NNX keeps the model object alive through training while still using
    JAX-compatible parameter containers and Optax optimizers. That makes the
    training code straightforward to read without sacrificing JAX-native
    behavior.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        *,
        out_features: int = 1,
        activation: ActivationName = "relu",
        init_scale: float = 1e-1,
        rngs: nnx.Rngs,
    ) -> None:
        """Construct a dense feed-forward network for tiled emulator inputs.

        Parameters
        ----------
        in_features:
            Width of the input feature vector. In practice this is the number
            of tiled axis coordinates plus the number of physical parameters.
        hidden_features:
            Width of each hidden layer.
        hidden_layers:
            Number of hidden layers before the final linear readout.
        out_features:
            Output width. The emulators currently use ``1`` because they
            predict one scalar target per tiled row.
        activation:
            Non-linearity applied after each hidden linear layer.
        init_scale:
            Standard deviation used for normal weight initialization.
        rngs:
            NNX random-number container used to initialize parameters.
        """
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.activation = activation
        self.init_scale = init_scale

        kernel_init = jax.nn.initializers.normal(stddev=init_scale)
        self.hidden = nnx.List()

        # The first layer lifts tiled inputs into hidden space. Remaining
        # layers keep the width fixed so workflow configs can describe the
        # network compactly in terms of one repeated hidden width.
        if hidden_layers > 0:
            self.hidden.append(
                nnx.Linear(
                    in_features,
                    hidden_features,
                    kernel_init=kernel_init,
                    rngs=rngs,
                )
            )
            for _ in range(hidden_layers - 1):
                self.hidden.append(
                    nnx.Linear(
                        hidden_features,
                        hidden_features,
                        kernel_init=kernel_init,
                        rngs=rngs,
                    )
                )
            readout_in_features = hidden_features
        else:
            readout_in_features = in_features

        self.readout = nnx.Linear(
            readout_in_features,
            out_features,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Map a batch of tiled emulator features to predicted scalar values."""
        act_fn = _activation_fn(self.activation)
        x = inputs
        for layer in self.hidden:
            x = act_fn(layer(x))
        return self.readout(x)


def init_mlp(
    key: jax.Array,
    in_features: int,
    hidden_features: int,
    out_features: int = 1,
    hidden_layers: int = 2,
    activation: ActivationName = "relu",
    scale: float = 1e-1,
) -> DenseMLP:
    """Create an initialized NNX MLP ready for training or inference.

    This small helper keeps model construction consistent across tests, CLI
    smoke runs, and the shared trainer. The caller provides shape choices and a
    random seed; the helper returns a fully initialized live NNX module.
    """
    return DenseMLP(
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        activation=activation,
        init_scale=scale,
        rngs=nnx.Rngs(key),
    )


def forward_mlp(
    model: DenseMLP,
    inputs: jnp.ndarray,
    activation: ActivationName | None = None,
) -> jnp.ndarray:
    """Run a dense MLP forward pass using the provided NNX model.

    The extra ``activation`` argument is only a guard for call sites that still
    think in terms of a functional API. The model object itself is the real
    source of truth for the network configuration.
    """
    if activation is not None and activation != model.activation:
        raise ValueError(
            "Requested activation does not match the instantiated model. "
            f"Expected {model.activation!r}, received {activation!r}."
        )
    return model(inputs)
