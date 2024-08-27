import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = (
    "/home/antonio/dev/qdots_efficient/profiling/tmp/jax_cache"
)
os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"


import logging

import jax

logging.getLogger("jax").setLevel(logging.INFO)
logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)


jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

jax.config.update("jax_explain_cache_misses", True)


import jax.numpy as jnp

from qdots_qll.models.single_dot_weak_coupling_GAME import SingleDotWeakCouplingGAME

x = jnp.ones((16000, 16000))


@jax.jit
def fn1(y):
    return x + y


with jax.log_compiles():
    jax.profiler.start_trace("tmp/tensorboard")
    a = fn1(jnp.array(1.0))
    a.block_until_ready()
    jax.profiler.stop_trace()
# model = SingleDotWeakCouplingGAME()


# @jax.jit
# def f_to_compile(particles_locations, time, model):
#     # jax.vmap(model.likelihood_particle, in_axes=(0, None))(particles_locations, time)
#     # jax.vmap(model.likelihood_particle, in_axes=(0, None))(particles_locations, time)
#     # jax.vmap(model.likelihood_particle, in_axes=(0, None))(particles_locations, time)

#     return jax.vmap(model.likelihood_particle, in_axes=(0, None))(
#         particles_locations, time
#     )


# key = jax.random.key(seed=0)
# pars_locations = jax.random.normal(key, shape=(100000, 4))


# # jax.profiler.start_trace("tmp/tensorboard")
# probs = f_to_compile(pars_locations, time=jnp.array([31.4]), model=model)
# # probs.block_until_ready()
# # jax.profiler.stop_trace()

# print("SECOND")
# probs = f_to_compile(pars_locations, time=jnp.array([31.4]), model=model)
# # probs.block_until_ready()
