import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/home/antonio/dev/qdots_efficient/jax-cache"
os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"


import logging
import jax

logging.getLogger("jax").setLevel(logging.INFO)
logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)


jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_explain_cache_misses", True)
