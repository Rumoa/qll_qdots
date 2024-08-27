import warnings
from abc import abstractmethod
from collections import namedtuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import host_callback
from jaxtyping import Array, Complex, Float, Int, Real
from tensorflow_probability.substrates import jax as tfp

from qdots_qll.distributions import Distribution, _est_cov, _est_mean
from qdots_qll.models.single_dot_weak_coupling_GAME import Data


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

    def resample(
        self,
        key,
        particles_locations,
        weights,
    ):
        no_particles = particles_locations.shape[0]
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

        new_particles_locations = jax.random.multivariate_normal(
            subkey, new_mu, sigma, shape=(no_particles,)
        )

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
                *(particle, subkey),
            )

        new_particles_locations = jax.vmap(resample_wrong_one, in_axes=(0, 0))(
            new_particles_locations, subkeys
        )

        new_weights = jnp.ones(no_particles) / no_particles
        return {
            "key": key,
            "weights": new_weights,
            "particles_locations": new_particles_locations,
        }


class Resampler(eqx.Module):
    @abstractmethod
    def resample(
        self, subkey: Array, distribution: Distribution, data: Data, *args, **kwargs
    ) -> Distribution:
        pass


class MetropolisSampler(Resampler):
    factor: float
    boundaries: Array
    model: eqx.Module

    def __init__(self, boundaries: Array, model: eqx.Module, factor=1):
        self.factor = factor
        self.boundaries = boundaries
        self.model = model

    @jax.jit
    def resample(
        self, subkey, distribution: Distribution, data, *args, **kwargs
    ) -> Distribution:
        key, subkey = jax.random.split(subkey)
        # multinomial sampling to introduce variability
        old_locations = self.multinomial_importance_sampling(subkey, distribution)

        key, subkey = jax.random.split(subkey)

        proposals = self.generate_proposals(
            subkey, distribution, locs_after_importance_sampling=old_locations
        )

        log_uniform = jnp.log(jax.random.uniform(subkey))

        old_log_lkl = self.model.batch_total_log_lkl(old_locations, data)
        new_log_lkl = self.model.batch_total_log_lkl(proposals, data)
        do_accept = log_uniform < new_log_lkl - old_log_lkl
        new_locations = jnp.where(do_accept[:, None], proposals, old_locations)

        no_particles = proposals.shape[0]
        dist_new_locs = Distribution(
            particles_locations=new_locations,
            weights=jnp.ones(no_particles) / no_particles,
        )
        return dist_new_locs

    def multinomial_importance_sampling(self, subkey, dist: Distribution):
        no_particles = dist.particles_locations.shape[0]

        new_locs = jax.random.choice(
            subkey,
            dist.particles_locations,
            shape=(no_particles,),
            p=dist.weights / dist.weights.sum(),
        )

        return new_locs

    def generate_proposals(
        self, subkey, original_dist: Distribution, locs_after_importance_sampling
    ) -> Array:
        cov = jnp.diag(original_dist.cov())
        return tfp.distributions.TruncatedNormal(
            loc=locs_after_importance_sampling,
            scale=jnp.sqrt(cov) * self.factor,
            low=self.boundaries[:, 0],
            high=self.boundaries[:, 1],
        ).sample(seed=subkey, sample_shape=1)[0, :, :]

    def acceptance_rates(self, proposals, old_locations, data):
        old_log_lkl = self.model.batch_total_log_lkl(old_locations, data)
        new_log_lkl = self.model.batch_total_log_lkl(proposals, data)

        acc_rates: Array = jnp.where(
            jnp.exp(old_log_lkl) > 0.0, jnp.exp(new_log_lkl - old_log_lkl), 1
        )
        acc_rates = self.filter_acc_rates(
            acc_rates, jnp.exp(new_log_lkl), jnp.exp(old_log_lkl)
        )
        return acc_rates

    def filter_acc_rates(self, acc_rates, new_likelihoods, old_likelihoods):
        acc_rates = jnp.where(
            jnp.logical_or(old_likelihoods == 0, acc_rates > 1),
            jnp.ceil(new_likelihoods),
            acc_rates,
        )
        return acc_rates

    def rejection_step(self, subkey, proposals, acc_rates, old_locs):
        """
        Probabilistically accept or reject the new samples.
        """
        accept: Array = jax.random.binomial(subkey, 1, acc_rates)
        new_locs = jnp.where(accept[:, None], proposals, old_locs)
        return new_locs


class LiuWestResampler(Resampler):
    boundaries: Array
    a: float
    # model: eqx.Module

    def __init__(self, boundaries: Array, a=0.98) -> None:
        self.a = a
        self.boundaries = boundaries
        # self.model = model

    def multinomial_importance_sampling(self, subkey, dist: Distribution) -> Array:
        no_particles = dist.particles_locations.shape[0]

        new_locs = jax.random.choice(
            subkey,
            dist.particles_locations,
            shape=(no_particles,),
            p=dist.weights / dist.weights.sum(),
        )

        return new_locs

    @jax.jit
    def resample(
        self, subkey, distribution: Distribution, *args, **kwargs
    ) -> Distribution:
        mu = distribution.ev()
        std_diag = jnp.sqrt(jnp.diag(distribution.cov()))
        a = self.a

        key, subkey = jax.random.split(subkey)
        locs_after_is = self.multinomial_importance_sampling(subkey, distribution)

        means = a * locs_after_is + (1 - a) * mu
        h = (1 - a**2) ** 0.5
        std_with_h = std_diag * h

        key, subkey = jax.random.split(key)
        new_locs = tfp.distributions.TruncatedNormal(
            loc=means,
            scale=std_with_h,
            low=self.boundaries[:, 0],
            high=self.boundaries[:, 1],
        ).sample(seed=subkey, sample_shape=1)[0, :, :]

        no_particles = distribution.no_particles
        dist_new_locs = Distribution(
            particles_locations=new_locs,
            weights=jnp.ones(no_particles) / no_particles,
        )
        return dist_new_locs
