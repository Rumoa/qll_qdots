from qdots_qll.models.models_scratch_for_drafting import BaseClassDimension
import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt

from jax.scipy.linalg import expm

from jax import jit

from jaxtyping import Array, Float, Complex, Int, Real

# Normalized identity+pauli matrices dim=2

# _G = jnp.array(
#     [
#         jnp.array(
#             [[1, 0], [0, 1]],
#         ),
#         jnp.array(
#             [[0, 1], [1, 0]],
#         ),
#         jnp.array(
#             [[0, -1j], [1j, 0]],
#         ),
#         jnp.array(
#             [[1, 0], [0, -1]],
#         ),
#     ]
# ) / jnp.sqrt(2)


@jit
def dag(A):
    return jnp.conjugate(A.T)
