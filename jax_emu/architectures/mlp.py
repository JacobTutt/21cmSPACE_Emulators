"""
MLP architecture for emulators with Flax NNX

This module provides a reusable class to construct, initialise and
call a simple dense MLP model. It allows you to configure:
- the input and output feature dimensions
- the number of hidden layers and their width
- the activation function used in the hidden layers
- the standard deviation of the weight initialisation
"""

from __future__ import annotations
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

# Activation Functions
# --------------------
# Provides a mapping from human-readable activation names to the actual
# JAX functions used in the model.
# - "relu":
# - "tanh":
# - "gelu":

ActivationName = Literal["relu", "tanh", "gelu"]


def _activation_fn(name: ActivationName):
    """
    Translate a human-readable activation name into the JAX callable used at runtime.
    """
    activations = {
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "gelu": jax.nn.gelu,
    }
    return activations[name]


# MLP Architecture
# ----------------

class DenseMLP(nnx.Module):
    """
    Dense MLP implemented with Flax NNX.

    This class initialises a feed-forward neural network and defines the forward pass through it.
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
        """
        Initialise an untrained dense feed-forward network for tiled emulator inputs.

        Parameters
        ----------
        in_features:
            Width of the input feature vector.
        hidden_features:
            Width of each hidden layer.
        hidden_layers:
            Number of hidden layers before the final linear readout.
        out_features:
            Output width.
        activation:
            Non-linearity applied after each hidden linear layer.
        init_scale:
            Standard deviation used for normal weight initialization.
        rngs:
            NNX random-number container used to initialize parameters.
        """
        # Store configuration parameters
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.activation = activation
        self.init_scale = init_scale

        # Initialise the JAX random number generator for weight initialisation.
        kernel_init = jax.nn.initializers.normal(stddev=init_scale)
        # Initialise the store of hidden layers as an NNX List.
        self.hidden = nnx.List()

        # Construct the Neural Network
            # Layer 1: Input Features to Hidden Layer 1
        if hidden_layers > 0:
            self.hidden.append(
                nnx.Linear(
                    in_features,
                    hidden_features,
                    kernel_init=kernel_init,
                    rngs=rngs,
                )
            )
            # Layer 2: Hidden Layer N-1 to Hidden Layer N
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
            # If no hidden layers, connect input directly to output
            readout_in_features = in_features
        # Final Layer: Hidden Layer N (or Input features) to Output Features
        self.readout = nnx.Linear(
            readout_in_features,
            out_features,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Neural Network Forward Pass.

        At each layer the linear transform is applied to the inputs from the previous layer,
        followed by the non-linear activation function.

        Parameters
        ----------
        inputs:
            A batch of input features with shape (batch_size, in_features).

        Returns
        -------
            A batch of output predictions with shape (batch_size, out_features).
        """
        # Set the activation function based on the configuration parameter
        act_fn = _activation_fn(self.activation)
        # Forward pass through the hidden layers
        x = inputs
        # For each hidden layer:
        # Apply the linear transformation followed by the activation function
        for layer in self.hidden:
            x = act_fn(layer(x))
        # Final readout layer (linear transformation without activation)
        return self.readout(x)
