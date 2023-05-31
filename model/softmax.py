from typing import Tuple

import numpy as np
from jax import jit
import jax.numpy as jnp


@jit
def model(X: jnp.array, W: jnp.array, b: jnp.array) -> jnp.array:
    """
    P(Y|X W b)

    Args:
        X: [N, D]
        W: [B, D, M]
        b: [B, M]

    Returns:
        [B, N, M]
    """
    non_linear = jnp.exp(jnp.matmul(X, W) + jnp.expand_dims(b, axis=1))  # [B, N, M]
    denominator = jnp.expand_dims(jnp.sum(non_linear, axis=2), axis=2)  # [B, N, 1]
    return non_linear / denominator  # [B, N, M]

### Bayesian Inference ###

def sample_theta(B: int, D: int, M: int, mu: float = 0, sigma: float = 1) -> Tuple[np.array, np.array]:
    """
    Samples W and b\n
    W ~ N(mu, sigma)\n
    b ~ N(mu, sigma)\n

    Args:
        B: batch size
        D: input space dimension
        M: number of class
        mu: mu of prior distribution
        sigma: sigma of prior distribution
    Returns:
        sampled W [B, D, M] and b [B, M]
    """
    W = np.random.normal(mu, sigma, size=(B, D, M))
    b = np.random.normal(mu, sigma, size=(B, M))
    return W, b

@jit
def sampled_posterior(X: jnp.array, Y: jnp.array, W: jnp.array, b: jnp.array) -> jnp.array:
    """
    proportionate to P(W b | X Y)\n
    P(Y|X W b) = P(y_1|x_1 W b) * ... * P(y_N|x_N W b)\n

    Args:
        X: [N, D] train input data
        Y: [N] train output data
        W: [B, D, M] sampled weight
        b: [B, M] sampled bias
    
    Returns:
        [B] proportionate posterior
    """
    # predicted distribution of Y given X W and b
    predicted = model(X, W, b)  # [B, N, M]
    predicted = predicted[:, jnp.arange(predicted.shape[1]), Y]  # [B, N]
    return jnp.prod(predicted, axis=1)  # [B]

def infer(x: jnp.array, X: jnp.array, Y: jnp.array, B: int, M: int, mu: float, sigma: float) -> jnp.array:
    """
    Bayesian Inference of softmax

    Args:
        x: [N_test, D] to be inferred input data
        X: [N_train, D] train input data
        Y: [N_train] train output data
        B: batch size
        M: number of classes
        mu: mu of prior distribution
        sigma: sigma of prior distribution

    Returns:
        (dist, sample_dist)
        dist: [N_test, M] joint predicted distribution
        sample_dist: [N_test, B, M] predicted distribution for every sample
    """
    D = x.shape[1]

    # get sampled thetas
    # W [B, D, M]
    # b [B, M]
    W, b = sample_theta(B, D, M, mu, sigma)

    # compute sampled posterior distribution given sampled thetas
    posterior = sampled_posterior(X, Y, W, b)  # [B]

    # compute distribution for every sampled W
    prob_w = model(x, W, b)  # [B, N_test, M]

    # compute non-normalized probability distribution
    sample_dist = jnp.transpose(prob_w * posterior[:, None, None], axes=(1, 0, 2))  # [N_test, B, M]
    dist = jnp.sum(sample_dist, axis=1) # [N_test, M]

    # normalize sample distribution
    sample_denominator = jnp.clip(jnp.sum(sample_dist, axis=2), a_min=jnp.finfo(jnp.float32).tiny) # [N_test, B]
    sample_dist /= sample_denominator[:, :, None]  # [N_test, B, M]

    # normalize distribution
    dist_denominator = jnp.clip(jnp.sum(dist, axis=1), a_min=jnp.finfo(jnp.float32).tiny) # [N_test]
    dist /= dist_denominator[:, None]  # [N_test, M]

    return dist, sample_dist


@jit
def infer_with_sampled(x: jnp.array, X: jnp.array, Y: jnp.array, W: jnp.array, b: jnp.array) -> jnp.array:
    """
    Bayesian Inference of softmax

    Args:
        x: [N_test, D] to be inferred input data
        X: [N_train, D] train input data
        Y: [N_train] train output data
        W: [B, D, M] sampled weight
        b: [B, M] sampled bias

    Returns:
        (dist, sample_dist)
        dist: [N_test, M] joint predicted distribution
        sample_dist: [N_test, B, M] predicted distribution for every sample
    """

    # compute sampled posterior distribution given sampled thetas
    posterior = sampled_posterior(X, Y, W, b)  # [B]

    # compute distribution for every sampled W
    prob_w = model(x, W, b)  # [B, N_test, M]

    # compute non-normalized probability distribution
    sample_dist = jnp.transpose(prob_w * posterior[:, None, None], axes=(1, 0, 2))  # [N_test, B, M]
    dist = jnp.sum(sample_dist, axis=1) # [N_test, M]

    # normalize sample distribution
    sample_denominator = jnp.clip(jnp.sum(sample_dist, axis=2), a_min=jnp.finfo(jnp.float32).tiny) # [N_test, B]
    sample_dist /= sample_denominator[:, :, None]  # [N_test, B, M]

    # normalize distribution
    dist_denominator = jnp.clip(jnp.sum(dist, axis=1), a_min=jnp.finfo(jnp.float32).tiny) # [N_test]
    dist /= dist_denominator[:, None]  # [N_test, M]

    return dist, sample_dist
