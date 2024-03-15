# Here we write the functions associated to the smc update.
# TODO: Make things cleaner bc it looks a bit sloppy


import jax
from jax import jit, vmap
import jax.numpy as jnp
from qdots_qll.models import game
import qdots_qll.all_funcs as all_f
from functools import partial
from qdots_qll.run import Run
import equinox as eqx

from qdots_qll.times_proposals import fim_time_generator

from qdots_qll.distributions import SimpleDistribution


from qdots_qll.distributions import ESS
from qdots_qll.distributions import est_mean, est_cov
from qdots_qll.run import Run

# @jit
# def _iteration_smc(
#     key,
#     iteration,
#     particles_locations,
#     weights,
#     model,
#     true_pars,
#     experimental_design_object,
#     initial_state,
#     number_of_experimental_repetitions=1,
# ):
#     key, subkey = jax.random.split(key)

#     # We need to substitute this with the function that computes guesses times,
#     # computes the fim for each one and update them.

#     # This estimate goes into the time generator function.

#     # t = fim_time_generator(
#     #     key=subkey,
#     #     estimated_particles=current_estimated_parameters,
#     #     model=model,
#     #     maxval=40,
#     # )

#     t = experimental_design_object.generate_time(
#         key=subkey,
#         particles_locations=particles_locations,
#         weights=weights,
#     )

#     # t = jax.random.uniform(key=subkey, minval=0.01, maxval=40)

#     keys = jax.random.split(key, number_of_experimental_repetitions + 1)

#     key = keys[0]

#     results = jax.vmap(model.generate_data, in_axes=(0, None, None, None))(
#         keys[1:], true_pars, t, initial_state
#     )

#     all_lkls = jax.vmap(model.likelihood_particle, in_axes=(0, None, None))(
#         particles_locations, t, initial_state
#     )

#     lkl_results_all_particles = vmap(all_f.likelihood_data, in_axes=(0, None))(
#         all_lkls, results
#     )

#     lkl = jnp.prod(lkl_results_all_particles, axis=1)

#     new_weights = all_f.update_weights(lkl, weights)


#     new_distribution = SimpleDistribution(particles_locations, new_weights)
#     # jax.debug.breakpoint()
#     # covariance = new_distribution.est_covariance(particles_locations, weights)
#     # estimates = new_distribution.est_mean(particles_locations, weights)
#     return (
#         key,
#         iteration+1,
#         particles_locations,
#         new_weights,
#         model,
#         true_pars,
#         experimental_design_object

#         new_distribution,
#         iteration + 1,
#         key,
#         t,  # We also return the time to save it later
#     )


# STOPPING FUNCTIONS: Used to check if we stop the while loop.


# @jit
# def _is_iter_lower_than_min_iter(iter, min_iterations):
#     """Returns True if iter < min_iterations

#     Args:
#         iter (_type_): _description_
#         min_iterations (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     return jax.lax.cond(
#         iter < min_iterations, lambda _: True, lambda _: False, None
#     )


# @jit
# def _std_check_exit(array_of_cov_norms, threshold):
#     """_summary_

#     Args:
#         array_of_cov_norms (_type_): Array with norms of covariance
#         threshold (_type_): threshold

#     Returns:
#         _type_: Returns true if the standard deviation among norms is
#         greater or equal than epsilon. False otherwise
#     """

#     pred = jnp.nanstd(array_of_cov_norms)

#     return jax.lax.cond(
#         (pred >= threshold),
#         lambda _: True,
#         lambda _: False,
#         None,
#     )


# @jit
# def _niter_check_exit(n_iter, max_iter):
#     """_summary_

#     Args:
#         a (_type_): Array with norms of covariance
#         threshold (_type_): threshold

#     Returns:
#         _type_: Returns true if the standard deviation among norms is
#                 greater or equal than epsilon. False otherwise
#     """

#     return jax.lax.cond(
#         n_iter < max_iter, lambda _: True, lambda _: False, None
#     )


# def _check_conditions_exit(run_object):
#     # array_of_cov_norms, n_iter, threshold_cov_norm, max_iter
#     """Returns True when execution should continue.
#        Three conditions are checked:
#     - True if n_iter (current iteration) is < than the minimum iteration
#         to start checking things.
#     - If n_iter > min_iter, returns True if n_iter is < max iterations
#     - If std_norm_cov_array of last 10 iterations  > than the threshold
#         to stop, return True.


#     Args:
#         array_of_cov_norms (_type_): _description_
#         n_iter (_type_): _description_
#         threshold_cov_norm (_type_): _description_
#         max_iter (_type_): _description_

#     Returns:
#         _type_: _description_
#     """

#     # From the object we extract important information. The current iteration,
#     # the max iteration, the min_iteration
#     # to start checking things and the threshold to stop.

#     n_iter = run_object.iteration
#     max_iter = run_object.max_iterations
#     min_iterations = run_object.min_iterations
#     threshold_cov_norm = run_object.std_threshold

#     # We select the last 10 covariances matrices.
#     full_cov_array = run_object.cov_array
#     # cov_array = run_object.cov_array[-10 + n_iter : n_iter]

#     lookback = 40

#     cov_array = jax.lax.dynamic_slice(
#         full_cov_array,
#         [n_iter - lookback] + list(full_cov_array.shape[-2:]),
#         [lookback] + list(full_cov_array.shape[-2:]),
#     )

#     # Compute the norm of these last covariances
#     array_of_cov_norms = jax.vmap(jnp.linalg.norm, in_axes=(0))(cov_array)

#     # We check the conditions regarding iter < max_iter and std of norm of
#     # covariance > threshold.

#     iter_condition = _niter_check_exit(n_iter, max_iter)
#     norm_std_condition = _std_check_exit(
#         array_of_cov_norms, threshold_cov_norm
#     )
#     # This first checks that iteration< min_iteration. Returns true.
#     # If false, check other conditions
#     return jax.lax.cond(
#         _is_iter_lower_than_min_iter(n_iter, min_iterations),
#         lambda _: True,
#         lambda a: jnp.logical_and(*a),
#         jnp.array([iter_condition, norm_std_condition]),
#     )


# @partial(jax.jit, static_argnums=1)
# def update_run_with_partial_func(run_object, partial_function_to_update):
#     # Reminder for myself of how it works.

#     # We call the update function with the parameters that are required.
#     # i.e: iteration, key, time, weights, particles_locations

#     # update the weights
#     updated_variables = partial_function_to_update(
#         run_object.unwrap_updatable_elements()
#     )  # this corresponds to iteration, key, times array,
#     # , weights, particles_location

#     # The updated variables are:
#     (
#         iteration,
#         key,
#         new_time,
#         weights,
#         particles_locations,
#         covariance,
#         estimates,
#     ) = updated_variables

#     # jax.debug.breakpoint()

#     # We decide whether to update or not
#     key, particles_locations, weights = jax.lax.cond(
#         all_f.ESS(weights) > weights.shape[0] / 2,
#         lambda a: a,
#         lambda a: all_f.resample_now(*a),
#         (key, particles_locations, weights),
#     )
#     # jax.debug.breakpoint()

#     new_iter = iteration  # takes the number of iteration to
#     # set covariance in correct place

#     # We need to add to the times array and to
#     # the weights and particle positions array.

#     old_cov_array = run_object.cov_array
#     new_cov_array = old_cov_array.at[new_iter].set(
#         covariance
#     )  # set of new covariance

#     # we repeat the same with the estimates

#     old_estimates_array = run_object.estimates_array
#     new_estimates_array = old_estimates_array.at[new_iter].set(
#         estimates
#     )  # set of new covariance

#     # set the new times
#     old_times_array = run_object.times_array
#     new_times_array = old_times_array.at[new_iter].set(new_time)

#     # we should set now the weights and the particle positions.

#     # We are going to do this more explicit.
#     # create another instance of run object with the updated things

#     return Run(
#         iteration,
#         key,
#         # new_time,
#         weights,
#         particles_locations,
#         new_cov_array,
#         new_estimates_array,
#         new_times_array,
#         # we need to add the weights array,
#         # The particles locations_array,
#         *run_object.unwrap_non_updatable_elements()
#     )

# return Run(
#     *(
#         updated_variables[:-2]
#     ),  # don't include the covariance vector and the estimated array
#     new_cov,  # add the updated array with covariance
#     new_estimates,
#     *run_object.unwrap_non_updatable_elements()
# )


# def _make_partial_update_smc_object(a, f):
#     return f(*a)


# def run_smc_loop(
#     initial_run,
#     func_update_smc,
#     model,
#     true_pars,
# ):
#     # This is the partial function that sets the true parameters and the model
#     # of the function that updates one smc step
#     f_for_iteration_smc_partial = partial(
#         func_update_smc, model=model, true_pars=true_pars
#     )

#     # The function
#     _iteration_smc_partial = partial(
#         _make_partial_update_smc_object, f=f_for_iteration_smc_partial
#     )
#     body_fun_for_while = jax.jit(
#         partial(
#             update_run_with_partial_func,
#             partial_function_to_update=_iteration_smc_partial,
#         )
#     )
#     return jax.lax.while_loop(
#         _check_conditions_exit, body_fun_for_while, initial_run
#     )


# example_update_cl(
#     example_update_cl(ecl, _iteration_smc_partial_fori),
#     _iteration_smc_partial_fori,
# ).cov_array


@jit
def update_array_at(index, arr, new_element):
    new_arr = arr.at[index].set(new_element)
    return new_arr


class SMCUpdater(eqx.Module):
    model: eqx.Module
    exp_design: eqx.Module
    resampler: eqx.Module
    initial_state: jax.Array
    true_pars: jax.Array
    number_exp_rep_matrix_shape: jax.Array

    def __init__(
        self,
        model,
        exp_design,
        resampler,
        initial_state,
        true_pars,
        number_exp_repetitions,
    ):
        self.model = model
        self.exp_design = exp_design
        self.resampler = resampler
        self.initial_state = initial_state
        self.true_pars = true_pars
        self.number_exp_rep_matrix_shape = jnp.identity(number_exp_repetitions)

    def likelihood_data(self, lkl_results, result):
        return lkl_results[result]

    def update_weights(self, lkl, weights):
        new_weights = lkl * weights
        new_weights = new_weights / jnp.sum(new_weights)
        return new_weights

    # @jit
    def _iteration_smc_new_weights(
        self, key, iteration, particles_locations, weights, *args, **kwargs
    ):
        key, subkey = jax.random.split(key)

        t = self.exp_design.generate_time(
            key=subkey,
            particles_locations=particles_locations,
            weights=weights,
        )
        number_of_exp_re = self.number_exp_rep_matrix_shape.shape[0]
        keys = jax.random.split(key, number_of_exp_re + 1)

        key = keys[0]

        results = jax.vmap(
            self.model.generate_data, in_axes=(0, None, None, None)
        )(keys[1:], self.true_pars, t, self.initial_state)

        all_lkls = jax.vmap(
            self.model.likelihood_particle, in_axes=(0, None, None)
        )(particles_locations, t, self.initial_state)

        lkl_results_all_particles = vmap(
            self.likelihood_data, in_axes=(0, None)
        )(all_lkls, results)

        lkl = jnp.prod(lkl_results_all_particles, axis=1)

        new_weights = self.update_weights(lkl, weights)
        # return key, iteration + 1, particles_locations, new_weights

        return {
            "iteration": iteration + 1,
            "key": key,
            "weights": new_weights,
            "particles_locations": particles_locations,
            "time": t,
            # self.cov_array,
        }

    @jit
    def _step_force_resampling(self, run_obj):

        updated_variables_bf_resampling = self._iteration_smc_new_weights(
            **run_obj.return_mutable_attributes()
        )

        updated_variables = jax.lax.cond(
            False,
            lambda a: a,
            lambda a: {**a, **self.resampler.resample(**a)},
            updated_variables_bf_resampling,
        )

        index_iter = updated_variables["iteration"]
        new_cov_array = update_array_at(
            index_iter, run_obj.cov_array, est_cov(**updated_variables)
        )
        new_estimates_array = update_array_at(
            index_iter, run_obj.estimates_array, est_mean(**updated_variables)
        )
        new_times_array = update_array_at(
            index_iter, run_obj.times_array, updated_variables["time"]
        )

        return Run(
            **{
                **updated_variables,
                **run_obj.return_immutable_attributes(),
                **{
                    "cov_array": new_cov_array,
                    "estimates_array": new_estimates_array,
                    "times_array": new_times_array,
                },
            }
        )

    @jit
    def step(self, run_obj):

        updated_variables_bf_resampling = self._iteration_smc_new_weights(
            **run_obj.return_mutable_attributes()
        )

        updated_variables = jax.lax.cond(
            ESS(updated_variables_bf_resampling["weights"])
            > updated_variables_bf_resampling["weights"].shape[0] / 2,
            lambda a: a,
            lambda a: {**a, **self.resampler.resample(**a)},
            updated_variables_bf_resampling,
        )

        index_iter = updated_variables["iteration"]
        new_cov_array = update_array_at(
            index_iter, run_obj.cov_array, est_cov(**updated_variables)
        )
        new_estimates_array = update_array_at(
            index_iter, run_obj.estimates_array, est_mean(**updated_variables)
        )
        new_times_array = update_array_at(
            index_iter, run_obj.times_array, updated_variables["time"]
        )

        return Run(
            **{
                **updated_variables,
                **run_obj.return_immutable_attributes(),
                **{
                    "cov_array": new_cov_array,
                    "estimates_array": new_estimates_array,
                    "times_array": new_times_array,
                },
            }
        )

    # @jit
    # def smc_run(self, stop_checker_function, initial_run):
    #     step_fun = self.step
    #     return jax.lax.while_loop(stop_checker_function, step_fun, initial_run)


@jax.jit
def SMC_run(
    initial_run,
    checker_obj,
    smc_obj,
):
    return jax.lax.while_loop(
        checker_obj.check_stop, smc_obj.step, initial_run
    )
