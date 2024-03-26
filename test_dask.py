from joblib import Parallel, delayed
import multiprocessing
import jax
import jax.numpy as jnp
import numpy as np

from joblib import parallel_config

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


with open("job.toml", "rb") as f:
    config = tomllib.load(f)


number_of_runs = int(config["run"]["number_of_runs"])
number_of_runs_compilation = int(
    config["run_for_compilation"]["number_of_runs"]
)

print(number_of_runs)
print(number_of_runs_compilation)

model = two_qdots_separable_maps(POVM_array=jnp.array(sic_povm(4)))

seed = config["run"]["seed"]


key = jax.random.PRNGKey(seed=seed)
key, subkey = jax.random.split(key)


resampler = LWResampler()
# exp_design = RandomExpDesign(0.01, 40)
# logging.info(f"Time optimizer: Determinant")
exp_design = MaxDetFimExpDesign(0.01, 40, 20, lr=0.5)
true_pars = jnp.array([0.314216, 0.35833, 0.053851, -0.333695])
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

key, subkey = jax.random.split(key)

keys_for_runs = jax.random.split(subkey, number_of_runs)


stopper_for_compilation = TerminationChecker(
    config["run_for_compilation"]["max_iterations"]
)

stopper_for_runs = TerminationChecker(config["run"]["max_iterations"])


f_SMC_run_compilation = lambda run: SMC_run(
    run,
    stopper_for_compilation,
    smcupdater,
)

run_example = (
    lambda key: initial_run_from_config(
        key,
        model,
        config["run_for_compilation"],
    )
)(keys_for_compilation[0])

print("Starting compilation")
jax.block_until_ready(f_SMC_run_compilation(run_example))
print("Compilation finished")


def f_parallel_runs(key):
    run = (
        lambda key: initial_run_from_config(
            key,
            model,
            config["run"],
        )
    )(key)

    result = (
        lambda run: SMC_run(
            run,
            stopper_for_runs,
            smcupdater,
        )
    )(run)
    return result


with parallel_config(backend="threading", n_jobs=80):
    result = Parallel()(
        delayed(f_parallel_runs)(i) for i in list(keys_for_runs)
    )


print(result)
exit()
# initial_runs_compilation = jax.vmap(
#     lambda key: initial_run_from_config(
#         key,
#         model,
#         config["run_for_compilation"],
#     )
# )(keys_for_compilation)

# print(initial_runs_compilation)

# stopper_for_compilation = TerminationChecker(
#     config["run_for_compilation"]["max_iterations"]
# )

# f_SMC_run_compilation = lambda run: SMC_run(
#     run,
#     stopper_for_compilation,
#     smcupdater,
# )


logging.info("Starting compilation runs")
# jax.block_until_ready(SMC_run_vmap_compilation(initial_runs_compilation))
logging.info("Compilation runs finished")
exit()


# ----------------------------------------------------------#


print(smcupdater)
f_initial_runs = lambda key: (
    initial_run_from_config(
        key,
        model,
        config["run_for_compilation"],
    )
)
with parallel_config(backend="threading", n_jobs=20):
    result = Parallel()(delayed(f_initial_runs)(i) for i in list(subkeys))


print(result)
