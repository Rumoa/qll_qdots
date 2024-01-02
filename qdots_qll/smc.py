# Here we write the functions associated to the smc update.
# TODO: Make things cleaner bc it looks a bit sloppy


import jax
from jax import jit, vmap
import jax.numpy as jnp
from qdots_qll.models import game
import qdots_qll.all_funcs as all_f
from functools import partial
from qdots_qll.run import Run


@jit
def _iteration_smc(
    iteration,
    key,
    weights,
    particles_locations,
    model,
    true_pars,
    number_of_experimental_repetitions=10,
):
    key, subkey = jax.random.split(key)
    t = jax.random.uniform(key=subkey, minval=0.01, maxval=40)
    keys = jax.random.split(key, number_of_experimental_repetitions + 1)

    key = keys[0]

    results = jax.vmap(model.generate_data, in_axes=(0, None, None))(
        keys[1:], true_pars, t
    )

    all_lkls = jax.vmap(model.likelihood_particle, in_axes=(0, None))(
        particles_locations, t
    )

    lkl_results_all_particles = vmap(game.likelihood_data, in_axes=(0, None))(
        all_lkls, results
    )

    lkl = jnp.prod(lkl_results_all_particles, axis=1)

    weights = all_f.update_weights(lkl, weights)
    covariance = all_f.est_cov(particles_locations, weights)
    estimates = all_f.est_mean(particles_locations, weights)
    return (
        iteration + 1,
        key,
        weights,
        particles_locations,
        covariance,
        estimates,
    )


# STOPPING FUNCTIONS: Used to check if we stop the while loop.


@jit
def _is_iter_lower_than_min_iter(iter, min_iterations):
    """Returns True if iter < min_iterations

    Args:
        iter (_type_): _description_
        min_iterations (_type_): _description_

    Returns:
        _type_: _description_
    """
    return jax.lax.cond(
        iter < min_iterations, lambda _: True, lambda _: False, None
    )


@jit
def _std_check_exit(array_of_cov_norms, threshold):
    """_summary_

    Args:
        array_of_cov_norms (_type_): Array with norms of covariance
        threshold (_type_): threshold

    Returns:
        _type_: Returns true if the standard deviation among norms is greater or equal than epsilon. False otherwise
    """

    pred = jnp.nanstd(array_of_cov_norms)

    return jax.lax.cond(
        (pred >= threshold),
        lambda _: True,
        lambda _: False,
        None,
    )


@jit
def _niter_check_exit(n_iter, max_iter):
    """_summary_

    Args:
        a (_type_): Array with norms of covariance
        threshold (_type_): threshold

    Returns:
        _type_: Returns true if the standard deviation among norms is greater or equal than epsilon. False otherwise
    """

    return jax.lax.cond(
        n_iter < max_iter, lambda _: True, lambda _: False, None
    )


def _check_conditions_exit(run_object):
    # array_of_cov_norms, n_iter, threshold_cov_norm, max_iter
    """Returns True when execution should continue. Three conditions are checked:
    - True if n_iter (current iteration) is < than the minimum iteration to start checking things.
    - If n_iter > min_iter, returns True if n_iter is < max iterations
    - If std_norm_cov_array of last 10 iterations  > than the threshold to stop, return True.


    Args:
        array_of_cov_norms (_type_): _description_
        n_iter (_type_): _description_
        threshold_cov_norm (_type_): _description_
        max_iter (_type_): _description_

    Returns:
        _type_: _description_
    """

    # From the object we extract important information. The current iteration, the max iteration, the min_iteration
    # to start checking things and the threshold to stop.

    n_iter = run_object.iteration
    max_iter = run_object.max_iterations
    min_iterations = run_object.min_iterations
    threshold_cov_norm = run_object.std_threshold

    # We select the last 10 covariances matrices.
    full_cov_array = run_object.cov_array
    # cov_array = run_object.cov_array[-10 + n_iter : n_iter]

    lookback = 40

    cov_array = jax.lax.dynamic_slice(
        full_cov_array,
        [n_iter - lookback] + list(full_cov_array.shape[-2:]),
        [lookback] + list(full_cov_array.shape[-2:]),
    )

    # Compute the norm of these last covariances
    array_of_cov_norms = jax.vmap(jnp.linalg.norm, in_axes=(0))(cov_array)

    # We check the conditions regarding iter < max_iter and std of norm of
    # covariance > threshold.

    iter_condition = _niter_check_exit(n_iter, max_iter)
    norm_std_condition = _std_check_exit(
        array_of_cov_norms, threshold_cov_norm
    )
    # This first checks that iteration< min_iteration. Returns true.
    # If false, check other conditions
    return jax.lax.cond(
        _is_iter_lower_than_min_iter(n_iter, min_iterations),
        lambda _: True,
        lambda a: jnp.logical_and(*a),
        jnp.array([iter_condition, norm_std_condition]),
    )


@partial(jax.jit, static_argnums=1)
def update_run_with_partial_func(run_object, partial_function_to_update):
    # update the weights
    updated_variables = partial_function_to_update(
        run_object.unwrap_updatable_elements()
    )

    (
        iteration,
        key,
        weights,
        particles_locations,
        covariance,
        estimates,
    ) = updated_variables

    key, particles_locations, weights = jax.lax.cond(
        all_f.ESS(weights) > weights.shape[0] / 2,
        lambda a: a,
        lambda a: all_f.resample_now(*a),
        (key, particles_locations, weights),
    )

    new_iter = updated_variables[
        0
    ]  # takes the number of iteration to set covariance in correct place

    old_cov = run_object.cov_array
    new_cov = old_cov.at[new_iter].set(
        updated_variables[4]
    )  # set of new covariance

    # we repeat the same with the estimates

    old_estimates = run_object.estimates_array
    new_estimates = old_estimates.at[new_iter].set(
        estimates
    )  # set of new covariance

    # create another instance of run object with the updated things
    return Run(
        *(
            updated_variables[:-2]
        ),  # don't include the covariance vector and the estimated array
        new_cov,  # add the updated array with covariance
        new_estimates,
        *run_object.unwrap_non_updatable_elements()
    )


def _make_partial_update_smc_object(a, f):
    return f(*a)


def run_smc_loop(
    initial_run,
    func_update_smc,
    model,
    true_pars,
):
    # This is the partial function that sets the true parameters and the model
    # of the function that updates one smc step
    f_for_iteration_smc_partial = partial(
        func_update_smc, model=model, true_pars=true_pars
    )

    # The function
    _iteration_smc_partial = partial(
        _make_partial_update_smc_object, f=f_for_iteration_smc_partial
    )
    body_fun_for_while = jax.jit(
        partial(
            update_run_with_partial_func,
            partial_function_to_update=_iteration_smc_partial,
        )
    )
    return jax.lax.while_loop(
        _check_conditions_exit, body_fun_for_while, initial_run
    )


# example_update_cl(
#     example_update_cl(ecl, _iteration_smc_partial_fori),
#     _iteration_smc_partial_fori,
# ).cov_array
