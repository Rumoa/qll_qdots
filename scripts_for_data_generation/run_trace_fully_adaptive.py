from datetime import datetime, timedelta
from time import process_time
from pathlib import Path
from qdots_qll.models.single_dot_weak_coupling_GAME import *

from qdots_qll.resamplers import LWResamplerBounds

from qdots_qll.exp_design import (
    OptimizeInitialStateMeasurements,
    MaxDetFimExpDesign,
    OptimizeInitialStateMeasurementsTrace,
    MaxTraceFimExpDesign,
)

from tensorflow_probability.substrates import jax as tfp

import joblib

from qdots_qll.distributions import (
    Distribution,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import numpy
from tensorflow_probability.substrates import jax as tfp

from qdots_qll.distributions import Distribution, update_weights

from qdots_qll.models.single_dot_weak_coupling_GAME import true_parameters


@eqx.filter_jit
def pls_resample(key, distribution):
    resample_result = resampler.resample(
        key, distribution.particles_locations, distribution.weights
    )
    key, new_weights, new_particles_locations = (
        resample_result["key"],
        resample_result["weights"],
        resample_result["particles_locations"],
    )
    distribution = Distribution(new_particles_locations, new_weights)
    return key, distribution


@eqx.filter_jit
def do_not_resample(key, distribution):
    return key, distribution


@jax.jit
def select_lkl_outcome(
    particle, outcome, t, prob_initial_state, prob_measurement_basis
):
    lkl = model.likelihood_particle_with_basis_initial_state(
        particle, t, prob_initial_state, prob_measurement_basis
    )[*outcome]
    return lkl


def initialize_carry(key):
    key, subkey = jax.random.split(key)
    particles_locations = tfp.distributions.TruncatedNormal(
        loc=mus, scale=sigmas, low=boundaries[:, 0], high=boundaries[:, 1]
    ).sample(seed=subkey, sample_shape=no_particles)

    weights = jnp.ones(no_particles) / no_particles

    pdist = Distribution(particles_locations, weights)
    return key, pdist, p_initial_state, p_measurement_basis


@jax.jit
def f_scan(carry, _):
    key, pdist, p_initial_state, p_measurement_basis = carry
    key, subkey = jax.random.split(key)

    t = jax.random.uniform(key=subkey, minval=0.01, maxval=50.0)
    # times_list.append(t)

    new_prob_initial_state, new_prob_measurement_basis = eqx.filter_jit(
        popt.optimize_probability_distribution
    )(
        dist_initial_state=p_initial_state,
        dist_measurement_basis=p_measurement_basis,
        model=model,
        t=t,
        particle=pdist.ev(),
    )
    key, subkey = jax.random.split(key)

    t = eqx.filter_jit(expdesign.generate_time)(
        key=subkey,
        particles_locations=pdist.particles_locations,
        weights=pdist.weights,
        model=model,
        prob_initial_state=new_prob_initial_state,
        prob_measurement_basis=new_prob_measurement_basis,
    )

    # In this case, we use the updated probability but we don't forward it to the next iteration

    key, subkey = jax.random.split(key)
    chosen_initial_state = jax.random.choice(
        subkey, jnp.arange(no_initial_states), p=new_prob_initial_state
    )

    key, subkey = jax.random.split(key)
    chosen_basis = jax.random.choice(
        subkey, jnp.arange(no_measurement_basis), p=new_prob_measurement_basis
    )

    key, subkey = jax.random.split(key)
    outcome = eqx.filter_jit(model.generate_data)(
        subkey, true_parameters, t, chosen_initial_state, chosen_basis
    )
    # outcomes_list.append(outcome)

    lkl_particles = jax.vmap(select_lkl_outcome, in_axes=(0, None, None, None, None))(
        pdist.particles_locations,
        outcome,
        t,
        new_prob_initial_state,
        new_prob_measurement_basis,
    )

    pdist = jax.jit(update_weights)(pdist, lkl_particles)

    key, pdist = jax.lax.cond(
        pdist.check_resampling(), pls_resample, do_not_resample, *(key, pdist)
    )

    return (key, pdist, new_prob_initial_state, new_prob_measurement_basis), (
        outcome,
        t,
        pdist.ev(),
        pdist.cov(),
        new_prob_initial_state,
        new_prob_measurement_basis,
    )


def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def transpose_results(pytree):
    return tree_stack(list(map(list, zip(*tree_unstack(tree_unstack(pytree))))))


init_time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
directory = Path("../results_one_qubit")
# directory = Path("ojo")

if not directory.exists():
    directory.mkdir(parents=True, exist_ok=True)

filename = str(directory) + str("/run_" + init_time)

# Definition of parameters

boundaries = jnp.array(
    [
        [0.1, 0.5],
        [0.1, 0.5],
        [0.01, 0.2],
        [-0.5, -0.01],
    ]
)

seed = 2
no_particles = 250

no_runs = 50
no_max_iterations = 10000

mus = boundaries.mean(axis=1)
sigmas = jnp.abs((boundaries[:, 0] - boundaries[:, 1]) / (2 * 1))

popt = OptimizeInitialStateMeasurementsTrace(iter=4, lr=0.05)
expdesign = MaxTraceFimExpDesign(t_min=0.01, t_max=45.0, sgd_iter=4, lr=0.01)
resampler = LWResamplerBounds(a=0.98, parameters_bounds=boundaries)

key = jax.random.PRNGKey(seed=seed)

model = SingleDotWeakCouplingGAME()

key, subkey = jax.random.split(key)
particles_locations = tfp.distributions.TruncatedNormal(
    loc=mus, scale=sigmas, low=boundaries[:, 0], high=boundaries[:, 1]
).sample(seed=subkey, sample_shape=no_particles)

weights = jnp.ones(no_particles) / no_particles

pdist = Distribution(particles_locations, weights)

no_initial_states = 4
no_measurement_basis = 3
p_initial_state = jnp.ones(no_initial_states) / no_initial_states
p_measurement_basis = jnp.ones(no_measurement_basis) / no_measurement_basis
new_p_initial_state, new_p_measurement_basis = p_initial_state, p_measurement_basis

f_scan_mapped = jax.vmap(f_scan, in_axes=(0, None))

initial_carries = jax.vmap(initialize_carry)(jax.random.split(subkey, no_runs))

print(f"Starting program {datetime.now()}")

print(f"Number of particles: {no_particles}")
print(f"Number of runs: {no_runs}")
print(f"Max iterations: {no_max_iterations}")

print(f"Estimating compilation and iteration time ...")
# Estimation of runtime
t1_start = process_time()
jax.lax.scan(f_scan_mapped, init=initial_carries, xs=None, length=1)
t1_stop = process_time()

comp_time = t1_stop - t1_start

t1_start = process_time()
jax.lax.scan(f_scan_mapped, init=initial_carries, xs=None, length=1)
t1_stop = process_time()

iter_time = t1_stop - t1_start

total_time = comp_time + iter_time * (no_max_iterations - 1)
final_time = datetime.now() + timedelta(seconds=total_time)
print(f"Compilation time: {comp_time} seconds")
print(f"Iteration time: {iter_time} seconds")
print(f"Expected total time: {total_time} seconds")
print(f"Expected finishing time: {final_time}")

print("Starting runs... ")
_, results = jax.lax.scan(
    f_scan_mapped, init=initial_carries, xs=None, length=no_max_iterations
)

joblib.dump(results, filename)
print(f"Runs completed at {datetime.now()}")
print(f"filename: {filename}")
