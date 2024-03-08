import jax
import jax.numpy as jnp
from jax import jit, vmap


# TODO Change this such that we can include different objective functions
@jit
def maximize_fim_time(t, particle, model, alpha):
    derivative_fim_to_t = jax.grad(
        (lambda par, t: (jnp.linalg.det(model.fim(par, t)))), 1
    )

    new_t = t

    grad_t = derivative_fim_to_t(particle, new_t)

    new_t = new_t + alpha * grad_t

    def body_fun_fori_loop(i, new_t):
        grad_t = derivative_fim_to_t(particle, new_t)
        new_t = new_t + alpha * grad_t
        return new_t

    return jax.lax.fori_loop(0, 20, body_fun_fori_loop, new_t)


# TODO Change this such that we can include different objective functions
@jit
def fim_time_generator(key, estimated_particles, model, maxval=100):
    no_candidates = 10
    key, subkey = jax.random.split(key)

    candidates = jax.random.uniform(
        subkey,
        shape=(no_candidates,),
        minval=0.01,
        maxval=maxval,
    )

    times = vmap(maximize_fim_time, in_axes=(0, None, None, None))(
        candidates, estimated_particles, model, 0.5
    )

    utilities = vmap(
        lambda time: (jnp.linalg.det(model.fim(estimated_particles, time))),
        in_axes=(0),
    )(times)
    return times[jnp.argmax(utilities)]
