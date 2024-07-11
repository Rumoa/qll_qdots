from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import jit

from qdots_qll.distributions import _est_mean
from qdots_qll.models.single_dot_weak_coupling_GAME import Experiment


class ExperimentalDesign(eqx.Module):
    @abstractmethod
    def generate_experiment(self, *args, **kwargs) -> Experiment:
        pass


class RandomExpDesign(eqx.Module):
    t_min: float
    t_max: float

    def __init__(self, t_min, t_max) -> None:
        self.t_min = t_min
        self.t_max = t_max

    @jit
    def generate_time(self, key, *args, **kwargs):
        return jax.random.uniform(key=key, minval=self.t_min, maxval=self.t_max)


class MaxDetFimExpDesign(eqx.Module):
    t_min: float
    t_max: float
    sgd_iter: int
    lr: float

    def __init__(self, t_min, t_max, sgd_iter, lr) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.sgd_iter = sgd_iter
        self.lr = lr

    # @jit
    def optimize_utility_function(self, model, t, **kwargs):
        loss_function = lambda t: -1 * jnp.linalg.det(model.fim(t=t, **kwargs))

        grad_f = jax.grad(loss_function)

        def f_for_scan(carry, _):
            params, opt_state = carry
            grad = grad_f(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return [params, opt_state], _

        solver = optax.adam(learning_rate=self.lr)
        params = t
        opt_state = solver.init(params)

        re, _ = jax.lax.scan(f_for_scan, [params, opt_state], None, length=3)
        return re[0]

    #
    @jit
    def generate_time(self, key, particles_locations, weights, model, **kwargs):
        est_particle = _est_mean(particles_locations, weights)

        util_fun = lambda t: 1 * jnp.linalg.det(
            model.fim(t=t, particle=est_particle, **kwargs)
        )

        no_candidates = 10

        key, subkey = jax.random.split(key)

        times_candidates = jax.random.uniform(
            subkey,
            shape=(no_candidates,),
            minval=self.t_min,
            maxval=self.t_max,
        )

        times_optimized = jax.vmap(
            lambda t: self.optimize_utility_function(
                model=model, t=t, particle=est_particle, **kwargs
            )
        )(times_candidates)

        utilities = jax.vmap(
            lambda t: (util_fun(t)),
        )(times_optimized)
        return times_optimized[jnp.argmax(utilities)]


class MaxTraceFimExpDesign(eqx.Module):
    t_min: float
    t_max: float
    sgd_iter: int
    lr: float

    def __init__(self, t_min, t_max, sgd_iter, lr) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.sgd_iter = sgd_iter
        self.lr = lr

    # @jit
    def optimize_utility_function(self, model, t, **kwargs):
        loss_function = lambda t: -1 * jnp.trace(model.fim(t=t, **kwargs))

        grad_f = jax.grad(loss_function)

        def f_for_scan(carry, _):
            params, opt_state = carry
            grad = grad_f(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return [params, opt_state], _

        solver = optax.adam(learning_rate=self.lr)
        params = t
        opt_state = solver.init(params)

        re, _ = jax.lax.scan(f_for_scan, [params, opt_state], None, length=3)
        return re[0]

    #
    @jit
    def generate_time(self, key, particles_locations, weights, model, **kwargs):
        est_particle = _est_mean(particles_locations, weights)

        util_fun = lambda t: 1 * jnp.trace(
            model.fim(t=t, particle=est_particle, **kwargs)
        )

        no_candidates = 10

        key, subkey = jax.random.split(key)

        times_candidates = jax.random.uniform(
            subkey,
            shape=(no_candidates,),
            minval=self.t_min,
            maxval=self.t_max,
        )

        times_optimized = jax.vmap(
            lambda t: self.optimize_utility_function(
                model=model, t=t, particle=est_particle, **kwargs
            )
        )(times_candidates)

        utilities = jax.vmap(
            lambda t: (util_fun(t)),
        )(times_optimized)
        return times_optimized[jnp.argmax(utilities)]


class OptimizeInitialStateMeasurements(eqx.Module):
    lr: int
    iter: int

    def __init__(self, lr=0.1, iter=5):
        self.lr = lr
        self.iter = iter

    def optimize_probability_distribution(
        self, dist_initial_state, dist_measurement_basis, **kwargs
    ):
        def loss_function(params, model, **kwargs):
            p_initial_state = params["state"]
            p_measurement_basis = params["measurement"]

            loss = -1 * jnp.linalg.det(
                model.fim(
                    prob_initial_state=p_initial_state,
                    prob_measurement_basis=p_measurement_basis,
                    **kwargs,
                )
            )
            return loss

        # optimizer = optax.sgd(learning_rate=self.lr)
        optimizer = optax.adam(learning_rate=self.lr)

        params = {
            "state": dist_initial_state,
            "measurement": dist_measurement_basis,
        }
        opt_state = optimizer.init(params)

        def f_for_scan(carry, _):
            params, opt_state = carry
            grad = jax.grad(loss_function)(params, **kwargs)
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            params["state"] = optax.projections.projection_simplex(params["state"])
            params["measurement"] = optax.projections.projection_simplex(
                params["measurement"]
            )

            return [params, opt_state], _

        re, _ = jax.lax.scan(f_for_scan, [params, opt_state], None, length=self.iter)
        return re[0]["state"], re[0]["measurement"]


class OptimizeInitialStateMeasurementsNoProjection(eqx.Module):
    lr: int
    iter: int

    def __init__(self, lr=0.1, iter=5):
        self.lr = lr
        self.iter = iter

    def optimize_probability_distribution(
        self, dist_initial_state, dist_measurement_basis, **kwargs
    ):
        def loss_function(params, model, **kwargs):
            p_initial_state = params["state"]
            p_measurement_basis = params["measurement"]

            loss = -1 * jnp.linalg.det(
                model.fim(
                    prob_initial_state=p_initial_state,
                    prob_measurement_basis=p_measurement_basis,
                    **kwargs,
                )
            )
            return loss

        # optimizer = optax.sgd(learning_rate=self.lr)
        optimizer = optax.adam(learning_rate=self.lr)

        params = {
            "state": dist_initial_state,
            "measurement": dist_measurement_basis,
        }
        opt_state = optimizer.init(params)

        def f_for_scan(carry, _):
            params, opt_state = carry
            grad = jax.grad(loss_function)(params, **kwargs)
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            # params["state"] = optax.projections.projection_simplex(
            #     params["state"]
            # )
            # params["measurement"] = optax.projections.projection_simplex(
            #     params["measurement"]
            # )

            return [params, opt_state], _

        re, _ = jax.lax.scan(f_for_scan, [params, opt_state], None, length=self.iter)
        return re[0]["state"], re[0]["measurement"]


class OptimizeInitialStateMeasurementsTrace(eqx.Module):
    lr: int
    iter: int

    def __init__(self, lr=0.1, iter=5):
        self.lr = lr
        self.iter = iter

    def optimize_probability_distribution(
        self, dist_initial_state, dist_measurement_basis, **kwargs
    ):
        def loss_function(params, model, **kwargs):
            p_initial_state = params["state"]
            p_measurement_basis = params["measurement"]

            loss = -1 * jnp.trace(
                model.fim(
                    prob_initial_state=p_initial_state,
                    prob_measurement_basis=p_measurement_basis,
                    **kwargs,
                )
            )
            return loss

        # optimizer = optax.sgd(learning_rate=self.lr)
        optimizer = optax.adam(learning_rate=self.lr)

        params = {
            "state": dist_initial_state,
            "measurement": dist_measurement_basis,
        }
        opt_state = optimizer.init(params)

        def f_for_scan(carry, _):
            params, opt_state = carry
            grad = jax.grad(loss_function)(params, **kwargs)
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            params["state"] = optax.projections.projection_simplex(params["state"])
            params["measurement"] = optax.projections.projection_simplex(
                params["measurement"]
            )

            return [params, opt_state], _

        re, _ = jax.lax.scan(f_for_scan, [params, opt_state], None, length=self.iter)
        return re[0]["state"], re[0]["measurement"]
