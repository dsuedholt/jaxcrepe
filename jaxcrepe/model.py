import functools
from typing import Literal

import jax
import jax.numpy as jnp
import equinox as eqx

import jaxcrepe


###########################################################################
# Model definition
###########################################################################


class Crepe(eqx.Module):
    """Crepe model definition"""
    
    conv1: eqx.nn.Conv2d
    conv1_BN: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    conv2_BN: eqx.nn.BatchNorm
    conv3: eqx.nn.Conv2d
    conv3_BN: eqx.nn.BatchNorm
    conv4: eqx.nn.Conv2d
    conv4_BN: eqx.nn.BatchNorm
    conv5: eqx.nn.Conv2d
    conv5_BN: eqx.nn.BatchNorm
    conv6: eqx.nn.Conv2d
    conv6_BN: eqx.nn.BatchNorm
    classifier: eqx.nn.Linear
    in_features: int

    def __init__(self, model: Literal['full', 'tiny'] = 'full', *, key):
        # Model-specific layer parameters
        if model == 'full':
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif model == 'tiny':
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        else:
            raise ValueError(f'Model {model} is not supported')

        # Shared layer parameters
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # Split keys for each layer
        keys = jax.random.split(key, 13)
        
        # Layer definitions
        self.conv1 = eqx.nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=0,
            key=keys[0])
        self.conv1_BN = eqx.nn.BatchNorm(
            input_size=out_channels[0],
            axis_name="batch",
            eps=0.0010000000474974513,
            momentum=0.99)  # JAX uses decay rate, PyTorch uses 1-decay

        self.conv2 = eqx.nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1],
            padding=0,
            key=keys[2])
        self.conv2_BN = eqx.nn.BatchNorm(
            input_size=out_channels[1],
            axis_name="batch",
            eps=0.0010000000474974513,
            momentum=0.99)

        self.conv3 = eqx.nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2],
            padding=0,
            key=keys[4])
        self.conv3_BN = eqx.nn.BatchNorm(
            input_size=out_channels[2],
            axis_name="batch",
            eps=0.0010000000474974513,
            momentum=0.99)

        self.conv4 = eqx.nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3],
            padding=0,
            key=keys[6])
        self.conv4_BN = eqx.nn.BatchNorm(
            input_size=out_channels[3],
            axis_name="batch",
            eps=0.0010000000474974513,
            momentum=0.99)

        self.conv5 = eqx.nn.Conv2d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4],
            padding=0,
            key=keys[8])
        self.conv5_BN = eqx.nn.BatchNorm(
            input_size=out_channels[4],
            axis_name="batch",
            eps=0.0010000000474974513,
            momentum=0.99)

        self.conv6 = eqx.nn.Conv2d(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5],
            padding=0,
            key=keys[10])
        self.conv6_BN = eqx.nn.BatchNorm(
            input_size=out_channels[5],
            axis_name="batch",
            eps=0.0010000000474974513,
            momentum=0.99)

        self.classifier = eqx.nn.Linear(
            in_features=self.in_features,
            out_features=jaxcrepe.PITCH_BINS,
            key=keys[12])

    def __call__(self, x, state, embed=False):
        # Forward pass through first five layers
        x, state = self.embed(x, state)

        if embed:
            return x, state

        # Forward pass through layer six
        x, state = self.layer(x, self.conv6, self.conv6_BN, state, (0, 0, 31, 32))

        # shape=(batch, self.in_features)
        x = x.transpose(0, 2, 1, 3).reshape(-1, self.in_features)

        # Compute logits
        return jax.nn.sigmoid(self.classifier(x)), state

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x, state):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]

        # Forward pass through first five layers
        x, state = self.layer(x, self.conv1, self.conv1_BN, state, (0, 0, 254, 254))
        x, state = self.layer(x, self.conv2, self.conv2_BN, state)
        x, state = self.layer(x, self.conv3, self.conv3_BN, state)
        x, state = self.layer(x, self.conv4, self.conv4_BN, state)
        x, state = self.layer(x, self.conv5, self.conv5_BN, state)

        return x, state

    def layer(self, x, conv, batch_norm, state, padding=(0, 0, 31, 32)):
        """Forward pass through one layer"""
        # Pad input
        x = jnp.pad(x, ((0, 0), (0, 0), (padding[2], padding[3]), (padding[0], padding[1])))
        
        # Convolution
        x = conv(x)
        
        # ReLU activation
        x = jax.nn.relu(x)
        
        # Batch normalization
        x, state = batch_norm(x, state)
        
        # Max pooling
        x = eqx.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))(x)
        
        return x, state


def create_model(model: Literal['full', 'tiny'] = 'full', *, key):
    """Create a CREPE model with initialized parameters."""
    return Crepe(model=model, key=key)


def inference(model, params, x, embed=False):
    """Run inference with a model, handling batch normalization state.
    
    This is a convenience function for inference that initializes
    the batch norm state and extracts only the predictions.
    """
    # Initialize batch norm state
    state = eqx.nn.State(model)
    
    # Run model
    output, _ = model(x, state, embed=embed)
    
    return output
