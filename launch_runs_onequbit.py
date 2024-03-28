import os
import multiprocessing

from qdots_qll.utils.povms import sigmas_povm
import jax
import numpy as np
import jax.numpy as jnp
import qutip as qt


import tomllib
from qdots_qll.exp_design import (
    RandomExpDesign,
    MaxDetFimExpDesign,
    MaxTraceFimExpDesign,
)
from qdots_qll.run import Run, initial_run_from_config
from qdots_qll.smc import SMCUpdater, SMC_run
from qdots_qll.resamplers import LWResampler
from qdots_qll.stop_conditions import TerminationChecker

from qdots_qll.distributions import (
    est_cov,
    est_mean,
    initialize_particle_locations,
    initialize_weights,
)
from qdots_qll.models.models_scratch_for_drafting import (
    SingleQDot3Params,
)
import joblib
import logging
from datetime import datetime
from qdots_qll.utils.generate_initial_state import max_entangled_dm_vec
from pprint import pformat, pprint
import argparse
from jax_tqdm import loop_tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument("--config", help="Config file of the job", dest="config")
args = parser.parse_args()

configfilename = args.config


init_time = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
run_filename = "results/run_" + init_time

logging.basicConfig(
    filename=run_filename + ".log",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - - %(levelname)s: %(message)s",
)


with open(configfilename, "rb") as f:
    config = tomllib.load(f)


logging.info(pformat(config["run"]))


number_of_runs = config["run"]["number_of_runs"]
number_of_runs_compilation = config["run_for_compilation"]["number_of_runs"]


ground_state_qdot = jnp.array(qt.ket2dm(qt.basis(2, 0))).flatten()
model = SingleQDot3Params(POVM_array=jnp.array(sigmas_povm))


seed = config["run"]["seed"]


key = jax.random.PRNGKey(seed=seed)


key, subkey = jax.random.split(key)


# print(initial_runs_compilation)
# keys = jax.random.split(key, number_of_runs)
true_pars = jnp.array(config["run"]["true_parameters"])


exp_design_dict = {
    "random": RandomExpDesign(0.01, 40),
    "maxdetfim": MaxDetFimExpDesign(0.01, 40, 20, lr=0.5),
    "maxtracefim": MaxTraceFimExpDesign(0.01, 40, 20, lr=0.5),
}

exp_design = exp_design_dict[config["run"]["exp_design"]]


resampler = LWResampler()


smcupdater = SMCUpdater(
    model=model,
    exp_design=exp_design,
    resampler=resampler,
    initial_state=ground_state_qdot,
    true_pars=true_pars,
    number_exp_repetitions=1,
)

stopper = TerminationChecker(config["run"]["max_iterations"])


keys = jax.random.split(subkey, number_of_runs)

initial_runs = (
    jax.vmap(
        lambda key: initial_run_from_config(
            key,
            model,
            config["run"],
        )
    )
)(keys)

n = config["run"]["max_iterations"]


@loop_tqdm(n)
def f_fori(i, r_obj):
    r_obj = smcupdater.step(r_obj)
    return r_obj


logging.info("Starting Runs")


results = jax.vmap(lambda run_0: jax.lax.fori_loop(0, n, f_fori, run_0))(
    initial_runs
)

joblib.dump(results, run_filename + "_results.job")
logging.info("Exiting")
