import jax
import jax.numpy as jnp

from jaxtyping import Array, Float, Complex, Int
import equinox as eqx


def est_mean(particles_locations, weights, **kwargs):
    return jnp.einsum("i, ij -> j", weights, particles_locations)


def est_cov(particles_locations, weights, **kwargs):
    return jnp.einsum(
        "i, im, ik -> mk", weights, particles_locations, particles_locations
    ) - jnp.einsum(
        "i, ij, m, mk -> jk",
        weights,
        particles_locations,
        weights,
        particles_locations,
    )


def ESS(weights):
    return 1 / jnp.sum(weights**2)


class Distribution(eqx.Module):
    no_particles: int
    no_parameters: int

    def __init__(self, no_particles: int, no_parameters: int) -> None:
        self.no_particles = no_particles
        self.no_parameters = no_parameters

    def est_mean(self, particles_locations, weights):
        return est_mean(particles_locations, weights)

    def est_covariance(self, particles_locations, weights):
        return est_cov(particles_locations, weights)


class SimpleDistribution(eqx.Module):
    no_particles: int
    no_parameters: int
    particles_locations: Array
    weights: Array

    def __init__(self, particles_locations, weights) -> None:
        self.no_particles = particles_locations.shape[0]
        self.no_parameters = particles_locations.shape[1]
        self.particles_locations = particles_locations
        self.weights = weights

    def est_mean(
        self,
    ):
        return est_mean(self.particles_locations, self.weights)

    def est_covariance(
        self,
    ):
        return est_cov(self.particles_locations, self.weights)
