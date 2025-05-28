<h1 align="center">jaxcrepe [WIP]</h1>
<div align="center">

</div>

Implementation of the CREPE [1] pitch tracker in JAX and Equinox, based on Max Morrison's
[PyTorch implementation](https://github.com/maxrmorrison/torchcrepe).

Current status: WIP. The core model code and the weights are ported, but
most of the CLI features are missing. If you take care of resampling audio to 16 kHz and
framing it into 1024-sample windows, you can use the model to get pitch probability distributions
like this:

```python
import jax
import jax.numpy as jnp
import jaxcrepe

jaxcrepe.load.model('full')

# single frame
w = 2 * jnp.pi * 440. / jaxcrepe.SAMPLE_RATE
frame = jnp.cos(w * jnp.arange(jaxcrepe.WINDOW_SIZE))
pitch_probs = jaxcrepe.infer.model(frame)

# batch of frames
w = 2 * jnp.pi * jnp.geomspace(220., 880., 3) / jaxcrepe.SAMPLE_RATE
frames = jnp.cos(w[:, None] * jnp.arange(jaxcrepe.WINDOW_SIZE)[None, :])
pitch_probs = jax.vmap(jaxcrepe.infer.model)(frames)
```

Decoding is not yet ported, but you can get rough `f0` values from the
pitch probabilities with a simple `argmax` operation:

```python
max_bins = jnp.argmax(pitch_probs, axis=-1)
f0s = jaxcrepe.convert.bins_to_frequency(max_bins)

print(f0s)  # [221.00099 438.45654 886.66187]
```

## Why?

Because JAX is awesome and blazing fast, but it needs more audio tooling. Simply porting the model code to Equinox
results in a 5-6x speedup on GPU, without any tricks - just using `jax.jit` and `jax.vmap`. Running `scripts/benchmark.py`
on an NVIDIA A5000 gives me:

```
torch: 62.6534 ms/it
jax: 11.1497 ms/it
```