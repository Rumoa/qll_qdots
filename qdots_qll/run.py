# This file contains the Run dataclass with the info that needs to be
# passed to the jitted functions.

import jax
import jax.numpy as jnp

import equinox as eqx


# # TODO: add fields to include the storage of MSE
# class Run(eqx.Module):
#     iteration: int
#     key: jax.Array
#     # time: jax.Array
#     weights: jax.Array
#     particles_locations: jax.Array
#     max_iterations: int
#     min_iterations: int
#     std_threshold: float
#     cov_array: jax.Array
#     estimates_array: jax.Array
#     times_array: jax.Array

#     def __init__(
#         self,
#         iteration,
#         key,
#         # time,
#         weights,
#         particles_locations,
#         cov_array,
#         estimates_array,
#         times_array,
#         max_iterations,
#         min_iterations,
#         std_threshold,
#     ):
#         self.iteration = iteration
#         self.key = key
#         # self.time = time
#         self.weights = weights
#         self.particles_locations = particles_locations
#         self.cov_array = cov_array
#         self.estimates_array = estimates_array
#         self.times_array = times_array
#         self.max_iterations = max_iterations
#         self.min_iterations = min_iterations
#         self.std_threshold = std_threshold

#     def unwrap_updatable_elements(self):
#         """
#         Used to extract the fields to be updated by the smc update step.
#         """
#         return [
#             self.iteration,
#             self.key,
#             self.weights,
#             self.particles_locations,
#             # self.cov_array,
#         ]

#     def unwrap_non_updatable_elements(self):
#         """
#         Complementary of the above function, except the cov_array
#         and estimates array.
#         """
#         return [
#             self.max_iterations,
#             self.min_iterations,
#             self.std_threshold,
#             # self.cov_array,
#         ]


class Run(eqx.Module):

    iteration: int
    key: jax.Array
    time: jax.Array
    weights: jax.Array
    particles_locations: jax.Array
    times_array: jax.Array
    estimates_array: jax.Array
    cov_array: jax.Array
    max_iterations: int
    min_iterations: int
    std_threshold: float

    def __init__(
        self,
        iteration,
        key,
        time,
        weights,
        particles_locations,
        cov_array,
        estimates_array,
        times_array,
        max_iterations,
        min_iterations,
        std_threshold,
    ):
        self.iteration = iteration
        self.key = key
        self.time = time
        self.weights = weights
        self.particles_locations = particles_locations
        self.cov_array = cov_array
        self.estimates_array = estimates_array
        self.times_array = times_array
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.std_threshold = std_threshold

    def return_mutable_attributes(self):
        """
        Used to extract the fields to be updated by the smc update step.
        """

        return {
            "iteration": self.iteration,
            "key": self.key,
            "weights": self.weights,
            "particles_locations": self.particles_locations,
            "cov_array": self.cov_array,
            "estimates_array": self.estimates_array,
            "times_array": self.times_array,
        }

    def return_immutable_attributes(self):
        """
        Complementary of the above function, except the cov_array
        and estimates array.
        """
        return {
            "max_iterations": self.max_iterations,
            "min_iterations": self.min_iterations,
            "std_threshold": self.std_threshold,
            # self.cov_array,
        }


from qdots_qll.distributions import (
    est_mean,
    est_cov,
    initialize_particle_locations,
    initialize_weights,
    initialize_particle_locations_normal_prior,
)


def initial_run_from_config(key, model, run_config_dictionary):
    key, subkey = jax.random.split(key)

    number_of_particles = run_config_dictionary["number_of_particles"]

    max_iterations = run_config_dictionary["max_iterations"]
    bnds = jnp.array(run_config_dictionary["pars_space_boundaries"])

    min_iterations = run_config_dictionary["min_iterations"]

    std_stop = run_config_dictionary["std_stop"]

    initial_cov_array = jnp.zeros(
        [
            max_iterations,
            model.number_of_parameters,
            model.number_of_parameters,
        ]
    )
    initial_times_array = jnp.zeros([max_iterations])
    # initial_particles_locations = initialize_particle_locations(
    #     subkey, model.number_of_parameters, number_of_particles, bnds
    # )  # this is the one we are gonna change now for priors

    initial_particles_locations = initialize_particle_locations_normal_prior(
        subkey,
        number_of_particles,
        bnds,
        sigmas=run_config_dictionary["sigmas_prior"],
    )  # this is the one we are gonna change now for priors

    initial_weights = initialize_weights(number_of_particles)
    initial_cov_array = (
        jnp.zeros(
            [
                max_iterations,
                model.number_of_parameters,
                model.number_of_parameters,
            ]
        )
        .at[0]
        .set(est_cov(initial_particles_locations, initial_weights))
    )
    initial_estimates_array = (
        jnp.zeros([max_iterations, model.number_of_parameters])
        .at[0]
        .set(est_mean(initial_particles_locations, initial_weights))
    )
    return Run(
        iteration=0,
        key=key,
        time=0,
        weights=initial_weights,
        particles_locations=initial_particles_locations,
        cov_array=initial_cov_array,
        estimates_array=initial_estimates_array,
        times_array=initial_times_array,
        max_iterations=max_iterations,  # max_iterations,
        min_iterations=min_iterations,
        std_threshold=std_stop,
    )
