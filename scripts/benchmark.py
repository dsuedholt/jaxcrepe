import numpy as np
import torch
import jax.numpy as jnp
import jax

import torchcrepe
import jaxcrepe

from time import perf_counter

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

torchcrepe.load.model(torch_device, 'full')
jaxcrepe.load.model('full')

batch_size = 256
win_size = torchcrepe.WINDOW_SIZE
noise = np.random.randn(batch_size, win_size)

n_warmup = 1
n_measured = 200

def timed(fn):
    for _ in range(n_warmup):
        fn()

    start = perf_counter()
    for _ in range(n_measured):
        fn()
    end = perf_counter()

    return (end - start) * 1000. / n_measured


torch_input = torch.tensor(noise, device=torch_device).to(torch.float32)
with torch.no_grad():
    torch_compiled = torch.compile(torchcrepe.infer.model)
    torch_time = timed(lambda: torch_compiled(torch_input))

jax_input = jnp.array(noise).astype(jnp.float32)
jax_compiled = jax.jit(jax.vmap(jaxcrepe.infer.model))
jax_time = timed(lambda: jax_compiled(jax_input))

print(f'torch: {torch_time:.4f} ms/it')
print(f'jax: {jax_time:.4f} ms/it')