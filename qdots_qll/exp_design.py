from abc import abstractmethod
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import jit
from jaxtyping import Array

from qdots_qll.experiments import Experiment, ExperimentSingleDotWeakCouplingGAME
from qdots_qll.distributions import _est_mean


def choose_from_dist(subkey, dist):
    dist = dist / dist.sum()
    outcome = jax.random.choice(subkey, len(dist), p=dist)
    return outcome


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

        util_fun = lambda t: 1 * jnp.linalg.det(  # noqa: E731
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
        def loss_function(t):
            return -1 * jnp.trace(model.fim(t=t, **kwargs))

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

        def util_fun(t):
            return 1 * jnp.trace(model.fim(t=t, particle=est_particle, **kwargs))

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


class RandExpDesignGAME(ExperimentalDesign):
    def __init__(
        self,
    ):
        pass

    def generate_experiment(
        self,
        subkey: Array,
        *args,
        **kwargs,
    ) -> Experiment:
        key, subkey = jax.random.split(subkey)
        # I am gonna generate a random time

        time = jax.random.uniform(subkey, minval=0.1, maxval=40.0)

        key, subkey = jax.random.split(key)

        chosen_initial_state = jax.random.choice(
            key,
            a=jnp.array([0, 1, 2, 3]),
        )

        chosen_measurement_basis = jax.random.choice(
            subkey,
            a=jnp.array([0, 1, 2]),
        )

        experiment = ExperimentSingleDotWeakCouplingGAME(
            t=time,
            initial_state=chosen_initial_state,
            measurement_basis=chosen_measurement_basis,
        )
        return experiment


@partial(jax.jit, static_argnames=["loss_function", "sgd_iters"])
def sgd_loop(loss_function, params, lr, sgd_iters, *args, **kwargs):
    def f_for_scan(carry, _):
        params, opt_state = carry
        grad = grad_f(params)
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return [params, opt_state], _

    grad_f = jax.grad(loss_function)
    solver = optax.adamw(learning_rate=lr)
    opt_state = solver.init(params)
    final_param, _ = jax.lax.scan(
        f_for_scan, init=[params, opt_state], length=sgd_iters
    )
    return final_param[0]


@partial(jax.jit, static_argnames=["loss_function", "sgd_iters"])
def sgd_loop_projection(loss_function, params, lr, sgd_iters, *args, **kwargs):
    def f_for_scan(carry, _):
        params, opt_state = carry
        grad = grad_f(params)
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        params["initial_state"] = optax.projections.projection_simplex(
            params["initial_state"]
        )
        params["measurement"] = optax.projections.projection_simplex(
            params["measurement"]
        )
        return [params, opt_state], _

    grad_f = jax.grad(loss_function)
    solver = optax.sgd(learning_rate=lr)
    opt_state = solver.init(params)
    final_param, _ = jax.lax.scan(
        f_for_scan, init=[params, opt_state], length=sgd_iters
    )
    return final_param[0]


class GameExpDesignTimeAdaptivity(ExperimentalDesign):
    t_min: float
    t_max: float
    sgd_iter: int
    lr: float
    no_time_candidates: int

    def __init__(self, t_min, t_max, sgd_iter=10, lr=0.001, no_time_candidates=10):
        self.t_min = t_min
        self.t_max = t_max
        self.sgd_iter = sgd_iter
        self.lr = lr
        self.no_time_candidates = no_time_candidates

    def loss_fn(
        self, estimated_particle, model, prob_initial_state, prob_measurement_basis
    ):
        return lambda t: -1 * jnp.linalg.det(
            model.fim(
                t=t,
                particle=estimated_particle,
                prob_initial_state=prob_initial_state,
                prob_measurement_basis=prob_measurement_basis,
            )
        )

    @eqx.filter_jit
    def generate_time(
        self,
        subkey,
        distribution,
        model,
        prob_initial_state,
        prob_measurement_basis,
        *args,
        **kwargs,
    ):
        estimated_particle = distribution.ev()
        loss = self.loss_fn(
            estimated_particle, model, prob_initial_state, prob_measurement_basis
        )

        times_candidates = jax.random.uniform(
            subkey,
            shape=(self.no_time_candidates,),
            minval=self.t_min,
            maxval=self.t_max,
        )

        times_optimized = jax.vmap(
            lambda init_time: sgd_loop(
                loss, init_time, lr=self.lr, sgd_iters=self.sgd_iter
            )
        )(times_candidates)

        utilities_candidates = jax.vmap(
            lambda t: (-1 * loss(t)),
        )(times_optimized)

        return times_candidates[jnp.argmax(utilities_candidates)]

    @eqx.filter_jit
    def generate_experiment(
        self,
        subkey,
        distribution,
        model,
        prob_initial_state,
        prob_measurement_basis,
        *args,
        **kwargs,
    ) -> Experiment:
        key, subkey = jax.random.split(subkey)
        time = self.generate_time(
            subkey,
            distribution,
            model,
            prob_initial_state,
            prob_measurement_basis,
            *args,
            **kwargs,
        )

        key, subkey = jax.random.split(key)

        initial_state = choose_from_dist(subkey, prob_initial_state)
        measurement_basis = choose_from_dist(key, prob_measurement_basis)

        experiment = ExperimentSingleDotWeakCouplingGAME(
            t=time, initial_state=initial_state, measurement_basis=measurement_basis
        )
        return experiment


class GameExpDesignFullAdaptivity(ExperimentalDesign):
    t_min: float
    t_max: float
    sgd_iter: int
    lr: float
    no_time_candidates: int

    def __init__(self, t_min, t_max, sgd_iter=5, lr=0.05, no_time_candidates=10):
        self.t_min = t_min
        self.t_max = t_max
        self.sgd_iter = sgd_iter
        self.lr = lr
        self.no_time_candidates = no_time_candidates

    def loss_fn(
        self,
        estimated_particle,
        model,
    ):
        def curried_loss(config_dict):
            t = config_dict["time"]
            prho0 = config_dict["initial_state"]
            pbasis = config_dict["measurement"]
            return -1 * jnp.linalg.det(
                model.fim(
                    t=t,
                    particle=estimated_particle,
                    prob_initial_state=prho0,
                    prob_measurement_basis=pbasis,
                )
            )

        return curried_loss

    @eqx.filter_jit
    def generate_time(
        self,
        subkey,
        distribution,
        model,
        prob_initial_state,
        prob_measurement_basis,
        *args,
        **kwargs,
    ):
        estimated_particle = distribution.ev()
        _aux_loss = self.loss_fn(estimated_particle, model)

        def loss(time):
            conf_dict = {
                "time": time,
                "initial_state": prob_initial_state,
                "measurement": prob_measurement_basis,
            }
            return _aux_loss(conf_dict)

        times_candidates = jax.random.uniform(
            subkey,
            shape=(self.no_time_candidates,),
            minval=self.t_min,
            maxval=self.t_max,
        )

        times_optimized = jax.vmap(
            lambda init_time: sgd_loop(
                loss, init_time, lr=self.lr, sgd_iters=self.sgd_iter
            )
        )(times_candidates)

        utilities_candidates = jax.vmap(
            lambda t: (-1 * loss(t)),
        )(times_optimized)

        return times_candidates[jnp.argmax(utilities_candidates)]

    @eqx.filter_jit
    def optimize_distributions(
        self,
        subkey,
        distribution,
        model,
        prob_initial_state,
        prob_measurement_basis,
        *args,
        **kwargs,
    ):
        estimated_particle = distribution.ev()
        loss = self.loss_fn(
            estimated_particle,
            model,
        )

        time_candidate = jax.random.uniform(
            subkey,
            minval=self.t_min,
            maxval=self.t_max,
        )

        init_params = {
            "time": time_candidate,
            "initial_state": prob_initial_state,
            "measurement": prob_measurement_basis,
        }

        optimized_params = sgd_loop_projection(
            loss, init_params, self.lr, self.sgd_iter
        )

        return (
            optimized_params["initial_state"],
            optimized_params["measurement"],
        )

    # @eqx.filter_jit
    def generate_experiment(
        self,
        subkey,
        distribution,
        model,
        prob_initial_state,
        prob_measurement_basis,
        *args,
        **kwargs,
    ):
        key, subkey = jax.random.split(subkey)

        # time, opt_pr_rho0, opt_pr_basis = self.generate_time_optimize_distributions(
        #     subkey, distribution, model, prob_initial_state, prob_measurement_basis
        # )

        opt_pr_rho0, opt_pr_basis = self.optimize_distributions(
            subkey, distribution, model, prob_initial_state, prob_measurement_basis
        )

        time = self.generate_time(
            subkey, distribution, model, prob_initial_state, prob_measurement_basis
        )

        key, subkey = jax.random.split(key)

        initial_state = choose_from_dist(subkey, opt_pr_rho0)
        measurement_basis = choose_from_dist(key, opt_pr_basis)

        experiment = ExperimentSingleDotWeakCouplingGAME(
            t=time, initial_state=initial_state, measurement_basis=measurement_basis
        )
        return experiment, opt_pr_rho0, opt_pr_basis


class TraceGameExpDesignFullAdaptivity(GameExpDesignFullAdaptivity):
    t_min: float
    t_max: float
    sgd_iter: int
    lr: float
    no_time_candidates: int

    def __init__(self, t_min, t_max, sgd_iter=5, lr=0.05, no_time_candidates=10):
        super().__init__(t_min, t_max, sgd_iter, lr, no_time_candidates)
        # self.t_min = t_min
        # self.t_max = t_max
        # self.sgd_iter = sgd_iter
        # self.lr = lr
        # self.no_time_candidates = no_time_candidates

    def loss_fn(
        self,
        estimated_particle,
        model,
    ):
        def curried_loss(config_dict):
            t = config_dict["time"]
            prho0 = config_dict["initial_state"]
            pbasis = config_dict["measurement"]
            return -1 * jnp.linalg.trace(
                model.fim(
                    t=t,
                    particle=estimated_particle,
                    prob_initial_state=prho0,
                    prob_measurement_basis=pbasis,
                )
            )

        return curried_loss
