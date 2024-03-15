import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)


import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")


import matplotlib.pyplot as plt
from qbism import sic_povm
from jaxtyping import Array, Float, Complex, Int
import tomllib
import equinox as eqx
import functools
from typing import Optional
from itertools import product
from qdots_qll.exp_design import RandomExpDesign
from qdots_qll.run import Run, initial_run_from_config
from qdots_qll.smc import SMCUpdater, SMC_run
from qdots_qll.resamplers import LWResampler
from qdots_qll.stop_conditions import TerminationChecker
from qdots_qll.models.game import true_pars
from qdots_qll.distributions import (
    est_cov,
    est_mean,
    initialize_particle_locations,
    initialize_weights,
)
from qdots_qll.models.models_scratch_for_drafting import (
    two_qdots_separable_maps,
)
import joblib
import logging
from datetime import datetime
from qdots_qll.utils.generate_initial_state import max_entangled_dm_vec
from pprint import pformat


init_time = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
run_filename = "results/run_" + init_time

logging.basicConfig(
    filename=run_filename + ".log",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - - %(levelname)s: %(message)s",
)


with open("job.toml", "rb") as f:
    config = tomllib.load(f)


logging.info(pformat(config["run"]))

no_cores = len(jax.devices())

logging.info(f"Number of cores {no_cores}")


number_of_runs = int(config["run"]["number_of_runs"] / no_cores) * no_cores
number_of_runs_compilation = (
    int(config["run_for_compilation"]["number_of_runs"] / no_cores) * no_cores
)


model = two_qdots_separable_maps(POVM_array=jnp.array(sic_povm(4)))

seed = config["run"]["seed"]


key = jax.random.PRNGKey(seed=seed)


key, subkey = jax.random.split(key)


# print(initial_runs_compilation)
# keys = jax.random.split(key, number_of_runs)


resampler = LWResampler()
exp_design = RandomExpDesign(0.01, 40)
smcupdater = SMCUpdater(
    model=model,
    exp_design=exp_design,
    resampler=resampler,
    initial_state=max_entangled_dm_vec,
    true_pars=true_pars,
    number_exp_repetitions=1,
)

# ----------------------------------------------------------#
keys_for_compilation = jax.random.split(
    subkey, number_of_runs_compilation
).reshape(no_cores, -1, 2)

initial_runs_compilation = jax.pmap(
    jax.vmap(
        lambda key: initial_run_from_config(
            key,
            model,
            config["run_for_compilation"],
        )
    )
)(keys_for_compilation)


stopper_for_compilation = TerminationChecker(
    config["run_for_compilation"]["max_iterations"]
)

SMC_run_vmap_compilation = jax.pmap(
    jax.vmap(
        lambda run: SMC_run(
            run,
            stopper_for_compilation,
            smcupdater,
        )
    )
)


logging.info(f"Starting compilation runs")
SMC_run_vmap_compilation(initial_runs_compilation)
logging.info(f"Compilation runs finished")


# ----------------------------------------------------------#

key, subkey = jax.random.split(key)

keys = jax.random.split(subkey, number_of_runs).reshape(no_cores, -1, 2)

stopper = TerminationChecker(config["run"]["max_iterations"])


SMC_run_vmap = jax.pmap(
    jax.vmap(
        lambda run: SMC_run(
            run,
            stopper,
            smcupdater,
        )
    )
)

initial_runs = jax.pmap(
    jax.vmap(
        lambda key: initial_run_from_config(
            key,
            model,
            config["run"],
        )
    )
)(keys)

logging.info(f"Starting Runs")


results = SMC_run_vmap(initial_runs)
logging.info(f"Runs finished")
logging.info(f"Saving results...")


joblib.dump(results, run_filename + "_results.job")
logging.info(f"Exiting")
