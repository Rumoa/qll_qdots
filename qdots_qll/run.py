# This file contains the Run dataclass with the info that needs to be
# passed to the jitted functions.

import jax

import equinox as eqx


# TODO: add fields to include the storage of MSE
class Run(eqx.Module):
    iteration: int
    key: jax.Array
    weights: jax.Array
    particles_locations: jax.Array
    max_iterations: int
    min_iterations: int
    std_threshold: float
    cov_array: jax.Array

    def __init__(
        self,
        iteration,
        key,
        weights,
        particles_locations,
        cov_array,
        max_iterations,
        min_iterations,
        std_threshold,
    ):
        self.iteration = iteration
        self.key = key
        self.weights = weights
        self.particles_locations = particles_locations
        self.cov_array = cov_array
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.std_threshold = std_threshold

    def unwrap_updatable_elements(self):
        """
        Used to extract the fields to be updated by the smc update step.
        """
        return [
            self.iteration,
            self.key,
            self.weights,
            self.particles_locations,
            # self.cov_array,
        ]

    def unwrap_non_updatable_elements(self):
        """
        Complementary of the above function, except the cov_array.
        """
        return [
            self.max_iterations,
            self.min_iterations,
            self.std_threshold,
            # self.cov_array,
        ]
