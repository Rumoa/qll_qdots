import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow_probability.substrates import jax as tfp

import import_before_profile
from qdots_qll.distributions import (
    Distribution,
)
from qdots_qll.exp_design import RandExpDesignGAME
from qdots_qll.models.single_dot_weak_coupling_GAME import (
    SingleDotWeakCouplingGAME,
)
from data import Data
from experiments import ExperimentSingleDotWeakCouplingGAME
from qdots_qll.resamplers import LiuWestResampler, MetropolisSampler
from qdots_qll.smc import SMCUpdater, replace_single_datum, replace_single_exp

model = SingleDotWeakCouplingGAME()

key = jax.random.key(0)
key, subkey = jax.random.split(key)

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

N_max_exps = 10000

_aux_exps = jax.vmap(lambda e: ExperimentSingleDotWeakCouplingGAME(*e))(
    jnp.repeat(jnp.array([13.0, 0, 0])[None, :], N_max_exps, axis=0)
)

_aux_outcomes = jnp.ones(N_max_exps) * 0

aux_data = jax.vmap(lambda exp, outcome: Data(exp, outcome), in_axes=(0, 0))(
    _aux_exps, _aux_outcomes
)
