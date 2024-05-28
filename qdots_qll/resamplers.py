from qdots_qll.distributions import _est_mean, _est_cov
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Complex, Int, Real
from jax.experimental import host_callback
from collections import namedtuple
import warnings


def is_valid_particle_array_version(particle, boundaries):
    particle = jnp.atleast_2d(particle)

    lower_bounds = boundaries[:, 0]
    upper_bounds = boundaries[:, 1]

    within_lower = jnp.all(particle >= lower_bounds, axis=1)
    within_upper = jnp.all(particle <= upper_bounds, axis=1)

    return jnp.logical_and(within_lower, within_upper)


def is_valid_particle(particle, boundaries):
    particle = jnp.atleast_2d(particle)
    return jnp.bool(
        ((boundaries[:, 1] > particle) * (boundaries[:, 0] <= particle)).T.prod(axis=0)
    )[0]


class LWResampler(eqx.Module):
    a: int

    def __init__(self, parameters_bounds, a=0.98):
        self.a = a
        self.parameters_bounds = parameters_bounds

    def resample(self, key, particles_locations, weights, *args, **kwargs):
        no_particles = particles_locations.shape[0]
        # no_rv = particles_locations.shape[1]
        mu = _est_mean(particles_locations, weights)
        h = jnp.sqrt(1 - self.a**2)
        sigma = _est_cov(particles_locations, weights) * h**2

        # sigma = (
        #     jnp.diag(jnp.array([10, 10, 1, 1, 1]))
        #     @ sigma
        #     @ jnp.diag(jnp.array([10, 10, 1, 1, 1]))
        # )

        key, subkey = jax.random.split(key)
        new_mu = (
            self.a
            * jax.random.choice(
                subkey, particles_locations, shape=(no_particles,), p=weights
            )
            + (1 - self.a) * mu
        )

        key, subkey = jax.random.split(key)
        new_particles_location = jax.random.multivariate_normal(
            subkey, new_mu, sigma, shape=(no_particles,)
        )

        new_weights = jnp.ones(no_particles) / no_particles
        # return key, new_particles_location, new_weights
        return {
            "key": key,
            "weights": new_weights,
            "particles_locations": new_particles_location,
            # self.cov_array,
        }


class LWResamplerBounds(eqx.Module):
    a: int
    max_iterations: int
    parameters_bounds: Float[Array, "d d"]

    def __init__(self, parameters_bounds, a=0.98):
        self.a = a
        self.parameters_bounds = parameters_bounds
        self.max_iterations = 10

    def cond_keep_resampling(self, candidate, iteration):
        return jnp.logical_and(
            ~is_valid_particle(candidate, self.parameters_bounds),
            iteration <= self.max_iterations,
        )

    @jax.jit
    def resample_one_particle(self, key, particles_locations, weights, mu, sigma):
        key, candidate = self.propose_new_particle(
            key, particles_locations, weights, mu, sigma
        )

        iteration = 0

        def body_fun_while(args):
            key, candidate, iteration = args

            key, new_candidate = self.propose_new_particle(
                key, particles_locations, weights, mu, sigma
            )

            return key, new_candidate, iteration + 1

        def cond_fun_while(args):
            key, candidate, iteration = args
            return self.cond_keep_resampling(candidate, iteration)

        key, candidate, iteration = jax.lax.while_loop(
            cond_fun_while, body_fun_while, (key, candidate, iteration)
        )

        return key, candidate, iteration

    def propose_new_particle(self, key, particles_locations, weights, mu, sigma):
        key, subkey = jax.random.split(key)
        new_mu_particle = (
            self.a
            * jax.random.choice(subkey, particles_locations, shape=(1,), p=weights)
            + (1 - self.a) * mu
        )
        key, subkey = jax.random.split(key)
        new_particle_location = jax.random.multivariate_normal(
            subkey, new_mu_particle, sigma, shape=(1,)
        )
        return key, new_particle_location

    def resample(self, key, particles_locations, weights, *args, **kwargs):
        no_particles = particles_locations.shape[0]
        no_rv = particles_locations.shape[1]
        mu = _est_mean(particles_locations, weights)
        h = jnp.sqrt(1 - self.a**2)
        sigma = _est_cov(particles_locations, weights) * h**2

        key, subkey = jax.random.split(key)
        new_mu = (
            self.a
            * jax.random.choice(
                subkey, particles_locations, shape=(no_particles,), p=weights
            )
            + (1 - self.a) * mu
        )

        key, subkey = jax.random.split(key)

        # subkeys = jax.random.split(subkey, no_particles)

        new_particles_locations = jax.random.multivariate_normal(
            subkey, new_mu, sigma, shape=(no_particles,)
        )

        # return key, new_particles_locations

        # array_is_valid = is_valid_particle_array_version(
        #     new_particles_locations, self.parameters_bounds
        # )
        #
        # number_wrong_particles = (~array_is_valid.flatten()).sum()

        key, subkey = jax.random.split(key)

        subkeys = jax.random.split(subkey, new_particles_locations.shape[0])

        def resample_wrong_one(particle, subkey):
            true_fun = lambda particle, subkey: particle

            def false_fun(particle, subkey):
                _, candidate, _ = self.resample_one_particle(
                    subkey, particles_locations, weights, mu, sigma
                )
                return candidate[0]

            return jax.lax.cond(
                is_valid_particle(particle, self.parameters_bounds),
                true_fun,
                false_fun,
                *(particle, subkey)
            )

        new_particles_locations = jax.vmap(resample_wrong_one, in_axes=(0, 0))(
            new_particles_locations, subkeys
        )

        # return key, new_particles_locations
        # new_particles_locations = new_particles_locations.reshape(no_particles, no_rv)

        # key, subkey = jax.random.split(key)
        # new_particles_location = jax.random.multivariate_normal(
        #     subkey, new_mu, sigma, shape=(no_particles,)
        # )

        # Now we need to check if the new particles are correct.

        # is_valid_particle(new_particles_location, self.parameters_bounds)

        new_weights = jnp.ones(no_particles) / no_particles
        # return key, new_particles_location, new_weights
        return {
            "key": key,
            "weights": new_weights,
            "particles_locations": new_particles_locations,
            # self.cov_array,
        }
