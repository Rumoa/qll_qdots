from qdots_qll.models import game
from qdots_qll.run import Run
import qdots_qll.all_funcs as all_f
from qdots_qll.smc import run_smc_loop, _iteration_smc
from jax import vmap
import jax.numpy as jnp
import jax

# import matplotlib.pyplot as plt
from datetime import datetime
import joblib

import logging

init_time = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")

run_filename = "results/run_" + init_time

logging.basicConfig(
    filename=run_filename + ".log",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - - %(levelname)s: %(message)s",
)

logging.info(f"We optimize with respect to determinant. We use Choi ")

logging.info(f"Max time now is 40 instead of 100 ")
seed = 10
logging.info(f"Initial seed: {seed}")
key = jax.random.PRNGKey(seed=seed)

key, subkey = jax.random.split(key)


model = game.physical_model()

true_pars = game.true_pars
bnds = jnp.array(
    [
        [0.01, 0.9],
        [0.01, 0.9],
        [0.001, 0.22],
        [-0.01, -0.9],
    ]
)


number_of_particles = 1000
logging.info(f"Number of particles: {number_of_particles}")

max_iterations = 1000000
min_iterations = 100
logging.info(f"Maximum iterations: {max_iterations}")
logging.info(f"Minimum iterations: {min_iterations}")

number_of_runs = 10

logging.info(f"Number of runs: {number_of_runs}")

std_stop = 1e-8

logging.info(f"Std threshold: {std_stop}")

keys = jax.random.split(key, number_of_runs)


def fun_to_parallelize_run_smc(key):
    key, subkey = jax.random.split(key)

    init_particles_locations = all_f.initialize_particle_locations(
        subkey, model.number_of_parameters, number_of_particles, bnds
    )
    init_weights = all_f.initialize_weights(number_of_particles)

    initial_cov_array = jnp.zeros(
        [
            max_iterations,
            model.number_of_parameters,
            model.number_of_parameters,
        ]
    )

    initial_estimates_array = jnp.zeros(
        [max_iterations, model.number_of_parameters]
    )
    initial_times_array = jnp.zeros([max_iterations])

    initial_run = Run(
        iteration=0,
        key=key,
        weights=init_weights,
        particles_locations=init_particles_locations,
        cov_array=initial_cov_array,
        estimates_array=initial_estimates_array,
        times_array=initial_times_array,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        std_threshold=std_stop,
    )

    result = run_smc_loop(
        initial_run=initial_run,
        func_update_smc=_iteration_smc,
        model=model,
        true_pars=true_pars,
    )
    return result


logging.info("Vmap is called")
results = vmap(fun_to_parallelize_run_smc, in_axes=(0))(keys)

logging.info("Saving model and results...")
joblib.dump(model, run_filename + "_model.job")
joblib.dump(results, run_filename + "_results.job")

logging.info("Runs completed")
