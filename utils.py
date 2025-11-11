import jax
import pickle
import jax.numpy as jnp
from config import Config
from flax import linen as nn
from typing import Callable, Tuple

cfg = Config()


# ------------------------------- RNN cell -------------------------------
class RNNCell(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: Callable

    @nn.compact
    def __call__(self,
                 st_pre: jnp.ndarray,
                 x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        U = self.param('U', nn.initializers.xavier_uniform(), (self.input_dim, self.hidden_dim))
        V = self.param('V', nn.initializers.orthogonal(), (self.hidden_dim, self.output_dim))
        W = self.param('W', nn.initializers.orthogonal(), (self.hidden_dim, self.hidden_dim))
        ba = self.param('ba', nn.initializers.zeros, (self.hidden_dim,))
        by = self.param('by', nn.initializers.zeros, (self.output_dim,))

        st = self.activation(jnp.dot(x, U) + jnp.dot(st_pre, W) + ba)
        ot = self.activation(jnp.dot(st, V) + by)
        return st, ot


# ------------------------------- Special character -------------------------------
sc_dict = {'<PAD>': 0}


# ------------------------------- FC layer -------------------------------
class FullyConnection(nn.Module):
    output_dim: int
    dropout_rate: float = cfg.dropout_rate

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 train: bool) -> jnp.ndarray:
        _, _, embed_dim = x.shape
        x = nn.Dense(embed_dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim)(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        return x


# ------------------------------- Causal self-attention -------------------------------
def attention(Q, K, V, mask):
    b, _, seq_len, d_k = Q.shape
    score = jnp.matmul(Q, jnp.transpose(K, (0, 1, 3, 2)))
    score /= jnp.sqrt(d_k)
    score = jnp.where(mask, -float('inf'), score)
    score = nn.softmax(score, axis=-1)
    score = jnp.matmul(score, V)
    score = jnp.transpose(score, (0, 2, 1, 3)).reshape(b, seq_len, -1)
    return score


class MultiHead(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self,
                 Q: jnp.ndarray,
                 K: jnp.ndarray,
                 V: jnp.ndarray,
                 mask: jnp.ndarray) -> jnp.ndarray:
        b, seq_len, embed_dim = Q.shape

        # Linear
        Q = nn.Dense(embed_dim)(Q)
        K = nn.Dense(embed_dim)(K)
        V = nn.Dense(embed_dim)(V)

        # Multi-head attention
        Q = jnp.transpose(Q.reshape(b, seq_len, self.num_heads, -1),
                          (0, 2, 1, 3))
        K = jnp.transpose(K.reshape(b, seq_len, self.num_heads, -1),
                          (0, 2, 1, 3))
        V = jnp.transpose(V.reshape(b, seq_len, self.num_heads, -1),
                          (0, 2, 1, 3))
        score = attention(Q, K, V, mask)

        # Linear
        score = nn.Dense(embed_dim)(score)
        return score


# ------------------------------- Embedding -------------------------------
def position_encoding(seq_len, embed_dim):
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, embed_dim, 2) * -(jnp.log(10000.0) / embed_dim))
    pe = jnp.zeros((seq_len, embed_dim))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe[None, :, :]


def pde_params_encoding(embed_dim, pde_params):
    x = jnp.stack(pde_params)
    x = nn.Dense(embed_dim)(x)
    return x


class Embedding(nn.Module):
    group_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.group_size,),
            strides=(self.group_size,),
            padding='VALID',
            use_bias=True,
        )(x)
        x = nn.gelu(x)
        return x


# ------------------------------- Training -------------------------------
def get_mini_batch(data, batch_size):
    s, _, _ = data.shape
    batches = []
    for i in range(s // batch_size):
        batch = data[i:(i + batch_size), :, :]
        batches.append(batch)
    return jnp.array(batches)


def divide_training_testing_set(key, dataset, n_training_samples, n_testing_samples,
                                training_samples_idx, testing_samples_idx):
    n_per_wave_number = n_training_samples // len(training_samples_idx)

    combinations = dataset.get_available_combinations()

    training_data = {}
    for i in training_samples_idx:
        lambdas, wave_numbers = dataset.get_training_data(wave_number_idx=i)
        lambdas_subset = lambdas[:n_per_wave_number]
        wave_numbers_subset = wave_numbers[:n_per_wave_number]
        training_data[combinations[i]] = (lambdas_subset, wave_numbers_subset)

    candidate_testing_data = {}
    for i in testing_samples_idx:
        lambdas, wave_numbers = dataset.get_training_data(wave_number_idx=i)
        lambdas_subset = lambdas[:n_per_wave_number]
        wave_numbers_subset = wave_numbers[:n_per_wave_number]
        candidate_testing_data[combinations[i]] = (lambdas_subset, wave_numbers_subset)

    candidate_testing_keys = list(candidate_testing_data.keys())
    total = len(candidate_testing_keys)
    indices = jax.random.choice(key, total, (n_testing_samples,), replace=False)
    testing_keys = [candidate_testing_keys[i] for i in indices]

    testing_data = {}
    for k in testing_keys:
        lambdas, wave_numbers = candidate_testing_data[k]
        testing_data[k] = (lambdas[:1], wave_numbers[:1])
    return training_data, testing_data
