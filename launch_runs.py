import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import jax
import jax.numpy as jnp


from qbism import sic_povm
import tomllib
from qdots_qll.exp_design import RandomExpDesign, MaxDetFimExpDesign
from qdots_qll.run import Run, initial_run_from_config
from qdots_qll.smc import SMCUpdater, SMC_run
from qdots_qll.resamplers import LWResampler
from qdots_qll.stop_conditions import TerminationChecker

# from qdots_qll.models.game import true_pars
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


number_of_runs = config["run"]["number_of_runs"]
number_of_runs_compilation = config["run_for_compilation"]["number_of_runs"]

print(number_of_runs)
print(number_of_runs_compilation)


model = two_qdots_separable_maps(POVM_array=jnp.array(sic_povm(4)))

seed = config["run"]["seed"]


key = jax.random.PRNGKey(seed=seed)


key, subkey = jax.random.split(key)


# print(initial_runs_compilation)
# keys = jax.random.split(key, number_of_runs)
true_pars = jnp.array(config["run"]["true_parameters"])

resampler = LWResampler()
# exp_design = RandomExpDesign(0.01, 40)
# logging.info(f"Time optimizer: Determinant")
exp_design = MaxDetFimExpDesign(0.01, 40, 20, lr=0.5)


smcupdater = SMCUpdater(
    model=model,
    exp_design=exp_design,
    resampler=resampler,
    initial_state=max_entangled_dm_vec,
    true_pars=true_pars,
    number_exp_repetitions=1,
)


# ----------------------------------------------------------#
keys_for_compilation = jax.random.split(subkey, number_of_runs_compilation)

initial_runs_compilation = jax.vmap(
    lambda key: initial_run_from_config(
        key,
        model,
        config["run_for_compilation"],
    )
)(keys_for_compilation)


example_run = (
    lambda key: initial_run_from_config(
        key,
        model,
        config["run_for_compilation"],
    )
)(keys_for_compilation[0])

stopper_for_compilation = TerminationChecker(
    config["run_for_compilation"]["max_iterations"]
)
# run one to compile SMC_run


logging.info("Starting compilation of one run")

jax.block_until_ready(
    (
        lambda run: SMC_run(
            run,
            stopper_for_compilation,
            smcupdater,
        )
    )(example_run)
)

logging.info("Single compilation run finished")

SMC_run_vmap_compilation = jax.vmap(
    lambda run: SMC_run(
        run,
        stopper_for_compilation,
        smcupdater,
    )
)


logging.info("Starting compilation runs")
jax.block_until_ready(SMC_run_vmap_compilation(initial_runs_compilation))
logging.info("Compilation runs finished")
# exit()


# ----------------------------------------------------------#

key, subkey = jax.random.split(key)

keys = jax.random.split(subkey, number_of_runs)

stopper = TerminationChecker(config["run"]["max_iterations"])


SMC_run_vmap = jax.vmap(
    lambda run: SMC_run(
        run,
        stopper,
        smcupdater,
    )
)

initial_runs = (
    jax.vmap(
        lambda key: initial_run_from_config(
            key,
            model,
            config["run"],
        )
    )
)(keys)

logging.info("Starting Runs")


results = jax.block_until_ready(SMC_run_vmap(initial_runs))
logging.info("Runs finished")
logging.info("Saving results...")


joblib.dump(results, run_filename + "_results.job")
logging.info("Exiting")
