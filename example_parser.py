import os
import multiprocessing

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
#     multiprocessing.cpu_count()
# )

import jax
import jax.numpy as jnp


from qbism import sic_povm
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

import argparse

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

print(config)
