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

seed = 1
key = jax.random.PRNGKey(seed=seed)

key, subkey = jax.random.split(key)

subkeys = jax.random.split(subkey, 10)

model = two_qdots_separable_maps(POVM_array=jnp.array(sic_povm(4)))


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
