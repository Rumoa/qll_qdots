from qdots_qll.distributions import est_mean, est_cov
import jax
import jax.numpy as jnp
import equinox as eqx


class LWResampler(eqx.Module):
    a: int

    def __init__(self, a=0.98):
        self.a = a

    def resample(self, key, particles_locations, weights, *args, **kwargs):
        no_particles = particles_locations.shape[0]
        no_pars = particles_locations.shape[1]
        mu = est_mean(particles_locations, weights)
        h = jnp.sqrt(1 - self.a**2)
        sigma = est_cov(particles_locations, weights) * h**2

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
