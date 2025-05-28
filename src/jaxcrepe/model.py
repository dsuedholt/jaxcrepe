from typing import Literal

import jax
import jax.numpy as jnp
import equinox as eqx

import jaxcrepe

class StatelessBatchNorm(eqx.Module):
    """Simplified stateless inference-only batch normalization"""

    mean: jnp.ndarray
    variance: jnp.ndarray

    weight: jnp.ndarray
    bias: jnp.ndarray

    eps: float = 0.0010000000474974513

    def __init__(self, input_size: int):
        self.mean = jnp.zeros(input_size)
        self.variance = jnp.ones(input_size)

        self.weight = jnp.ones(input_size)
        self.bias = jnp.zeros(input_size)

    def __call__(self, x):
        def _norm(y, m, v, w, b):
            out = (y - m) / jnp.sqrt(v + self.eps)
            out = out * w + b
            return out

        out = jax.vmap(_norm)(x, self.mean, self.variance, self.weight, self.bias)
        return out

###########################################################################
# Model definition
###########################################################################

class Crepe(eqx.Module):
    """Crepe model definition"""
    
    conv1: eqx.nn.Conv2d
    conv1_BN: StatelessBatchNorm
    conv2: eqx.nn.Conv2d
    conv2_BN: StatelessBatchNorm
    conv3: eqx.nn.Conv2d
    conv3_BN: StatelessBatchNorm
    conv4: eqx.nn.Conv2d
    conv4_BN: StatelessBatchNorm
    conv5: eqx.nn.Conv2d
    conv5_BN: StatelessBatchNorm
    conv6: eqx.nn.Conv2d
    conv6_BN: StatelessBatchNorm
    classifier: eqx.nn.Linear
    in_features: int

    def __init__(self, model: Literal['full', 'tiny'] = 'full'):
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

        keys = jax.random.split(jax.random.PRNGKey(0), 7)

        batch_norm_fn = StatelessBatchNorm

        # Layer definitions
        self.conv1 = eqx.nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=0,
            key=keys[0]
        )

        self.conv1_BN = batch_norm_fn(input_size=out_channels[0])

        self.conv2 = eqx.nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1],
            padding=0,
            key=keys[1]
        )

        self.conv2_BN = batch_norm_fn(input_size=out_channels[1])

        self.conv3 = eqx.nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2],
            padding=0,
            key=keys[2])

        self.conv3_BN = batch_norm_fn(input_size=out_channels[2])

        self.conv4 = eqx.nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3],
            padding=0,
            key=keys[3])

        self.conv4_BN = batch_norm_fn(input_size=out_channels[3])

        self.conv5 = eqx.nn.Conv2d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4],
            padding=0,
            key=keys[4])

        self.conv5_BN = batch_norm_fn(input_size=out_channels[4])

        self.conv6 = eqx.nn.Conv2d(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5],
            padding=0,
            key=keys[5])

        self.conv6_BN = batch_norm_fn(input_size=out_channels[5])

        self.classifier = eqx.nn.Linear(
            in_features=self.in_features,
            out_features=jaxcrepe.PITCH_BINS,
            key=keys[6]
        )

    def __call__(self, x, embed=False):
        # Forward pass through first five layers
        x = self.embed(x)

        if embed:
            return x

        # Forward pass through layer six
        x = self.layer(x, self.conv6, self.conv6_BN, (0, 0, 31, 32))

        # shape=(self.in_features,)
        x = x.transpose(1, 0, 2).reshape((self.in_features,))

        # Compute logits
        return jax.nn.sigmoid(self.classifier(x))

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x):
        """Map input audio to pitch embedding"""
        # shape=(1, 1024, 1)
        x = x[None, :, None]

        # Forward pass through first five layers
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        x = self.layer(x, self.conv5, self.conv5_BN)

        return x

    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        """Forward pass through one layer"""
        # Pad input
        x = jnp.pad(x, ((0, 0), (padding[2], padding[3]), (padding[0], padding[1])))
        
        # Convolution
        x = conv(x)
        
        # ReLU activation
        x = jax.nn.relu(x)
        
        # Batch normalization
        x = batch_norm(x)
        
        # Max pooling
        x = eqx.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))(x)
        
        return x
