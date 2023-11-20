import jax.numpy as jnp
from jax import jit

# from jax import grad, jit, vmap, value_and_grad
# from jax import random

import jax

import numpy as np

# import matplotlib.pyplot as plt

from functools import partial


import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"


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
    Also clips the probability between zero and one, beware,
    it can fail silently.
    Args:
        rho_v (_type_): vectorized density matrix (n, 1)
        measurement_op (_type_): vectorized POVM operator (n, 1)

    Returns:
        _type_: _description_
    """
    praw = jnp.real(jnp.dot(jnp.conjugate(vec_rho).T, vec_measurement_op)).astype(
        float
    )[0, 0]

    upper_trim = jax.lax.cond(praw <= 1.0, lambda a: a, lambda a: 0.0, praw)
    lower_trim = jax.lax.cond(upper_trim >= 0.0, lambda a: a, lambda a: 0.0, upper_trim)

    return lower_trim
    # return jnp.clip(
    #     jnp.real(jnp.dot(jnp.conjugate(vec_rho).T, vec_measurement_op)),
    #     0.0,
    #     1.0,
    # )[0, 0]


@jit
def create_dij(particle, model):
    d = model["Identity matrix"].shape[0]
    dij = jnp.zeros([d**2 - 1, d**2 - 1], dtype=jnp.complex64)

    dij = dij.at[0, 2].set(particle[0] + 1j * particle[1])
    dij = dij.at[1, 2].set(particle[2] + 1j * particle[3])

    dij = dij + jnp.conjugate(dij.T)
    dij = dij.at[2, 2].set(particle[4])
    return dij


@jit
def dij_gg(dij, mat_gg):
    return jnp.einsum("ij, ijkl -> kl", dij, mat_gg[1:, 1:])


@jit
def make_H_renormalization(particles_h):
    return jnp.array(
        [
            [particles_h[0], particles_h[1] + 1j * particles_h[2]],
            [particles_h[1] - 1j * particles_h[2], 0],
        ]
    )


@jit
def make_liouvillian(H, model):
    d = model["Identity matrix"].shape[0]
    return 1j * jnp.kron(H.T, jnp.identity(d)) - 1j * jnp.kron(jnp.identity(d), H)


@jit
def generate_liouvillian_particle(particle, model):
    H0 = model["H"]
    mat_gg = model["Mat_gg"]
    dissipator = dij_gg(create_dij(particle[3:], model), mat_gg)
    H_renorm = make_H_renormalization(particle[0:3])
    H = H0 + H_renorm
    return make_liouvillian(H, model) + dissipator


@jit
def evolve_particle(particle, t, model):
    rho0 = model["Initial state"]
    L = generate_liouvillian_particle(particle, model)
    return v_evolve(t, L, rho0)


@jit
def p_E0_particle(particle, E_0, t, model):
    return compute_p(evolve_particle(particle, t, model), E_0)


@jit
def p_data_particle(data, particle, E_0, t, model):
    p = p_E0_particle(particle, E_0, t, model)

    return jax.lax.cond(data == 0, lambda: p, lambda: 1 - p)


@jit
def compute_likelihood_data(data, particles, E_0, t, model):
    return jnp.prod(
        jax.vmap(
            jax.vmap(
                p_data_particle,
                in_axes=(0, None, None, None, None),
            ),
            in_axes=(None, 0, None, None, None),
        )(data, particles, E_0, t, model),
        axis=1,
    )


@jit
def update_weights(lkl, weights):
    new_weights = lkl * weights
    new_weights = new_weights / jnp.sum(new_weights)
    return new_weights


@jit
def est_mean(particles_location, weights):
    return jnp.einsum("i, ij -> j", weights, particles_location)


@jit
def est_cov(particles_locations, weights):
    return jnp.einsum(
        "i, im, ik -> mk", weights, particles_locations, particles_locations
    ) - jnp.einsum(
        "i, ij, m, mk -> jk",
        weights,
        particles_locations,
        weights,
        particles_locations,
    )


def make_diss_gjgi(G, N=2):
    mat_gg = np.zeros([N**2, N**2, N**2, N**2], dtype=np.complex64)
    for i, gi in enumerate(list(G)):
        for j, gj in enumerate(list(G)):
            val = np.kron(np.transpose(gj), gi)
            val = val - 0.5 * np.kron(np.identity(2), gj @ gi)
            val = val - 0.5 * np.kron(np.transpose(gj @ gi), np.identity(2))
            mat_gg[i, j] = val
    return mat_gg


def initialize_weights(no_of_particles):
    N = no_of_particles
    return jnp.ones(N) / N


def populate_one_axis(key, bnds, no_particles):
    return jax.random.uniform(
        key, minval=jnp.min(bnds), maxval=jnp.max(bnds), shape=[no_particles]
    )


def initialize_particle_locations(key, model):
    no_of_parameters = model["Number of parameters"]
    boundaries = model["Space boundaries"]
    no_of_particles = model["Number of particles"]
    subkey = jax.random.split(key, no_of_parameters + 1)
    key = subkey[1]
    subkeys = subkey[1:]
    return (
        key,
        jax.vmap(populate_one_axis, in_axes=(0, 0, None))(
            subkeys, boundaries, no_of_particles
        ).T,
    )


# TODO write this in jax
# def spre(A):
#     return np.kron(np.eye(2), A)
#
# def spost(B):
#     return np.kron(B.T, np.eye(2))
#
# def sprepost(A,B):
#     return np.kron(B.T, A)
