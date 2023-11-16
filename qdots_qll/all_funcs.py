import jax.numpy as jnp
from jax import jit

# from jax import grad, jit, vmap, value_and_grad
# from jax import random

import jax

import numpy as np

# import matplotlib.pyplot as plt

# from functools import partial


# import os

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"


@jit
def vec(a):
    """Returns the vectorized version of a using the column ordering.

    Args:
        a (_type_): Operator with shape (n, n)

    Returns:
        _type_: Vectorized version of a with shape (n^2, 1)
    """
    return jnp.expand_dims(a.flatten("F"), 1)


@jit
def dag(a):
    """Returns the conjugate transpose of a

    Args:
        a (_type_): _description_

    Returns:
        _type_: a^{\\dagger}
    """
    return jnp.conjugate(a.T)


@jit
def v_trace(vec_a):
    """
    trace of vectorized  operator (n x n) a
    :param a: vectorized operator
    :return: a dot vec(I)
    """
    one = jnp.identity(int(np.sqrt(vec_a.shape[0])), dtype=jnp.complex64)

    return jnp.dot(dag(vec(one)), vec_a)


@jit
def v_evolve(t, lindbladian, vec_rho0):
    """Evolves the vectorized state given a Lindbladian and time

    Args:
        t (_type_): time
        lindbladian (_type_): shape(n, n) lindbladian
        vec_rho0 (_type_): vectorized initial state (n, 1)

    Returns:
        _type_: v = exp(t L) @ v0
    """
    rho = jax.scipy.linalg.expm(lindbladian * t) @ vec_rho0
    return rho / v_trace(rho)


@jit
def compute_p(vec_rho, vec_measurement_op):
    """Compute the probability of the vectorized state rho_v
    of obtaining the measurement associated to the measurement operator

    Args:
        rho_v (_type_): vectorized density matrix (n, 1)
        measurement_op (_type_): vectorized POVM operator

    Returns:
        _type_: _description_
    """
    return jnp.clip(jnp.real(jnp.dot(vec_rho, vec_measurement_op)), 0.0, 1.0)
