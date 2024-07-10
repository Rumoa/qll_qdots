import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int


def _est_mean(particles_locations, weights, **kwargs):
    return jnp.einsum("i, ij -> j", weights, particles_locations)


def _est_cov(particles_locations, weights, **kwargs):
    return jnp.einsum(
        "i, im, ik -> mk", weights, particles_locations, particles_locations
    ) - jnp.einsum(
        "i, ij, m, mk -> jk",
        weights,
        particles_locations,
        weights,
        particles_locations,
    )


def _multiply_lkl(u, p):
    if u is None:
        return p
    else:
        return p * u


def _ESS(weights):
    return 1 / jnp.sum(weights**2)


def update_weights(dist, new_lkl):
    get_weights = lambda t: t.weights
    new_weights = dist.weights * new_lkl
    new_weights = new_weights / new_weights.sum()
    return eqx.tree_at(where=get_weights, pytree=dist, replace=new_weights)


def update_particles_locations(dist, new_particles_locations):
    get_particles_locations = lambda t: t.particles_locations
    return eqx.tree_at(
        where=get_particles_locations, pytree=dist, replace=new_particles_locations
    )


class Distribution(eqx.Module):
    no_rv: int
    no_particles: int
    particles_locations: Array
    weights: Array

    def __init__(self, particles_locations, weights) -> None:
        self.no_particles = particles_locations.shape[0]
        self.no_rv = particles_locations.shape[1]
        self.particles_locations = particles_locations
        self.weights = weights

    def ev(
        self,
    ):
        return _est_mean(
            particles_locations=self.particles_locations, weights=self.weights
        )

    def cov(
        self,
    ):
        return _est_cov(
            particles_locations=self.particles_locations, weights=self.weights
        )

    def ESS(self):
        return _ESS(weights=self.weights)

    def check_resampling(self, resampling_threshold=0.5) -> Array:
        return self.ESS() <= resampling_threshold * self.no_particles


def initialize_particle_locations(
    key,
    no_of_parameters,
    no_of_particles,
    boundaries,
):
    # no_of_parameters = model["Number of parameters"]
    # boundaries = model["Space boundaries"]
    # no_of_particles = model["Number of particles"]
    subkey = jax.random.split(key, no_of_parameters)
    # key = subkey[1]
    # subkeys = subkey[1:]
    return (
        jax.vmap(populate_one_axis, in_axes=(0, 0, None))(
            subkey, boundaries, no_of_particles
        )
    ).T


def initialize_particle_locations_normal_prior(
    subkey, no_of_particles, boundaries, sigmas=5
):
    covs = jnp.diagflat(jnp.std(boundaries, axis=1) / sigmas) ** 2
    mus = jnp.mean(boundaries, axis=1)
    return jax.random.multivariate_normal(
        subkey, mean=mus, cov=covs, shape=(no_of_particles,)
    )


def populate_one_axis(key, bnds, no_particles):
    return jax.random.uniform(
        key, minval=jnp.min(bnds), maxval=jnp.max(bnds), shape=[no_particles]
    )


def initialize_weights(no_of_particles):
    N = no_of_particles
    return jnp.ones(N) / N
