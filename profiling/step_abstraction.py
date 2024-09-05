import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = (
    "/home/antonio/dev/qdots_efficient/profiling/tmp/jax_cache"
)
os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"

import logging

logging.getLogger("jax").setLevel(logging.INFO)
logging.basicConfig(filename="step.log", encoding="utf-8", level=logging.DEBUG)

import jax
import jax.numpy as jnp

jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

jax.config.update("jax_explain_cache_misses", True)

from tensorflow_probability.substrates import jax as tfp

from qdots_qll.distributions import (
    Distribution,
)
from qdots_qll.exp_design import RandExpDesignGAME
from qdots_qll.models.single_dot_weak_coupling_GAME import (
    SingleDotWeakCouplingGAME,
)
from data import Data
from experiments import ExperimentSingleDotWeakCouplingGAME
from qdots_qll.resamplers import MetropolisSampler
from qdots_qll.smc import SMCUpdater

seed = 0
key = jax.random.key(seed=seed)
key, subkey = jax.random.split(key)

model = SingleDotWeakCouplingGAME()

boundaries = jnp.array(
    [
        [0.1, 0.5],
        [0.1, 0.5],
        [0.01, 0.2],
        [-0.5, -0.01],
    ]
)

loc = boundaries.mean(axis=1)
scale = boundaries.std(axis=1)

truncated_norm = tfp.distributions.TruncatedNormal(
    loc=loc, scale=scale / 1.5, low=boundaries[:, 0], high=boundaries[:, 1]
)

no_particles = 500
init_particles_locations = truncated_norm.sample(
    seed=subkey, sample_shape=(no_particles,)
)
weights = jnp.ones(no_particles) / no_particles

dist = Distribution(particles_locations=init_particles_locations, weights=weights)

exp_design = RandExpDesignGAME()
resampler = MetropolisSampler(boundaries=boundaries, model=model)
# resampler = LiuWestResampler(boundaries=boundaries)


smc = SMCUpdater(model=model, exp_design=exp_design, resampler=resampler)

key, subkey = jax.random.split(key)

fake_first_data = Data(ExperimentSingleDotWeakCouplingGAME(-9999, 9999, 99999), 9999)
data = fake_first_data

max_iterations = 10
iteration = 0

exp_design = RandExpDesignGAME()
resampler = MetropolisSampler(boundaries=boundaries, model=model)
# resampler = LiuWestResampler(boundaries=boundaries)


smc = SMCUpdater(model=model, exp_design=exp_design, resampler=resampler)

rmse_list = []
no_particles = int(500)
init_particles_locations = truncated_norm.sample(
    seed=subkey, sample_shape=(no_particles,)
)
weights = jnp.ones(no_particles) / no_particles

dist = Distribution(particles_locations=init_particles_locations, weights=weights)

key, subkey = jax.random.split(key)

fake_first_data = Data(ExperimentSingleDotWeakCouplingGAME(-9999, 9999, 99999), 9999)

data = fake_first_data

with jax.log_compiles():
    jax.profiler.start_trace("tmp/tensorboard")
    key, dist, data = smc.step(iteration=iteration, key=key, dist=dist, data=data)
    dist.particles_locations.block_until_ready()

    for _ in range(5):
        key, dist, data = smc.step(iteration=iteration, key=key, dist=dist, data=data)

        dist.particles_locations.block_until_ready()
        iteration = iteration + 1
    jax.profiler.stop_trace()
