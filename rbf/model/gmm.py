import jax
import jax.numpy as jnp
from typing import Tuple
from flax import linen as nn
from rbf_dataset_utility.model import generate_rbf_solutions


class GMM(nn.Module):
    K: int

    @nn.compact
    def __call__(self,
                 eval_points: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        # Trainable RBF parameters
        mus = self.param('mus', nn.initializers.uniform(scale=5), (self.K, 2))
        log_sigmas = self.param('log_sigmas', nn.initializers.uniform(scale=5), (self.K, 2))
        angles = self.param('angles', nn.initializers.uniform(scale=5), (self.K,))
        weights = self.param('weights', nn.initializers.uniform(scale=5), (self.K,))

        # Format as (K, 6) and expand to batch (1, K, 6)
        lambda_kernels = jnp.concatenate([
            mus,
            log_sigmas,
            angles[:, None],
            weights[:, None],
        ], axis=1)

        lambda_batch = lambda_kernels[None, :, :]

        # Evaluate RBF solution
        u = generate_rbf_solutions(eval_points, lambda_batch)[0]

        return u
