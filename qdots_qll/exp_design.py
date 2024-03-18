import jax
import jax.numpy as jnp
from jax import jit
import equinox as eqx
from qdots_qll.distributions import est_mean
import optax


class RandomExpDesign(eqx.Module):
    t_min: float
    t_max: float

    def __init__(self, t_min, t_max) -> None:
        self.t_min = t_min
        self.t_max = t_max

    @jit
    def generate_time(self, key, *args, **kwargs):
        return jax.random.uniform(
            key=key, minval=self.t_min, maxval=self.t_max
        )


class MaxDetFimExpDesign(eqx.Module):
    t_min: float
    t_max: float
    sgd_iter: int
    lr: float

    def __init__(self, t_min, t_max, sgd_iter, lr, *args, **kwargs) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.sgd_iter = sgd_iter
        self.lr = lr

    def utility_fun(self, model):
        return lambda *args, **kwargs: jnp.linalg.det(
            model.fim(*args, **kwargs)
        )

    @jit
    def optimize_utility_function(
        self, t, particle, model, initial_state, **kwargs
    ):
        def grad_f(t):
            return -jax.grad(self.utility_fun(model), 1)(
                particle, t, initial_state
            )

        # grad_f = lambda t: -jax.grad(self.utility_fun(model), 1)(
        #     particle, t, initial_state
        # )

        solver = optax.adam(learning_rate=self.lr)
        params = t
        opt_state = solver.init(params)

        for _ in range(10):
            grad = grad_f(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
        return params

    @jit
    def generate_time(
        self,
        key,
        particles_locations,
        weights,
        model,
        initial_state,
        *args,
        **kwargs
    ):
        estimated_particles = est_mean(particles_locations, weights)
        no_candidates = 15

        key, subkey = jax.random.split(key)

        times_candidates = jax.random.uniform(
            subkey,
            shape=(no_candidates,),
            minval=self.t_min,
            maxval=self.t_max,
        )

        times_optimized = jax.vmap(
            lambda t: self.optimize_utility_function(
                t, estimated_particles, model, initial_state
            )
        )(times_candidates)

        utilities = jax.vmap(
            lambda t: (
                self.utility_fun(model)(estimated_particles, t, initial_state)
            ),
        )(times_optimized)
        return times_optimized[jnp.argmax(utilities)]
