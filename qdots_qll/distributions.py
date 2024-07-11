import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int

from qdots_qll.utils.utils import ensure_array, ensure_particles_shape


class Distribution(eqx.Module):
    """

    Args:
        eqx (_type_): _description_

    Returns:
        _type_: _description_
    """

    no_rv: int
    no_particles: int
    particles_locations: Array
    log_weights: Array
    weights: Array
    ESS: Array

    def __init__(
        self,
        particles_locations: Array,
        weights: Array = None,
        log_weights: Array = None,
    ) -> None:
        self.particles_locations = ensure_particles_shape(particles_locations)
        self.no_particles = self.particles_locations.shape[0]
        self.no_rv = self.particles_locations.shape[1]

        if weights is not None and log_weights is None:
            self.log_weights = normalize_log_weights(jnp.log(weights))
        if weights is None and log_weights is not None:
            self.log_weights = normalize_log_weights(log_weights)

    @property
    def weights(self):
        return jnp.exp(self.log_weights)

    def ev(
        self,
    ) -> Array:
        return _est_mean(
            particles_locations=self.particles_locations, weights=(self.weights)
        )

    def cov(
        self,
    ) -> Array:
        return _est_cov(
            particles_locations=self.particles_locations, weights=(self.weights)
        )

    @property
    def ESS(self) -> float:
        return _ESSlog(logweights=self.log_weights)

    def check_resampling(self, resampling_threshold=0.5) -> Array:
        return self.ESS <= resampling_threshold * self.no_particles


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


def _ESSlog(logweights: Array) -> Array:
    return 1 / jnp.sum(jnp.exp(2 * logweights))


def normalize_log_weights(logweights: Array) -> Array:
    new_logweights = logweights - jax.scipy.special.logsumexp(logweights)
    return new_logweights


def update_log_weights(dist: Distribution, new_log_lkl: Array):
    get_log_weights = lambda logdist: logdist.log_weights
    new_log_weights = dist.log_weights + new_log_lkl
    new_log_weights = normalize_log_weights(new_log_weights)
    return eqx.tree_at(where=get_log_weights, pytree=dist, replace=new_log_weights)


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
