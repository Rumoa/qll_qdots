import jax
import jax.numpy as jnp

from qdots_qll.models.single_dot_weak_coupling_GAME import SingleDotWeakCouplingGAME

model = SingleDotWeakCouplingGAME()

times = jnp.linspace(0, 40.0, 1000)

jax.profiler.start_trace("tmp/tensorboard")
probs = jax.vmap(lambda t: model.likelihood_particle(model.true_parameters, t))(times)

probs.block_until_ready()
# jax.profiler.stop_trace()


# jax.profiler.start_trace("tmp/tensorboard")
probs = jax.vmap(
    lambda t: model.likelihood_particle(model.true_parameters * 1.00001, t)
)(times)

probs.block_until_ready()

jax.profiler.stop_trace()
