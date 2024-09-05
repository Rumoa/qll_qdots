import equinox as eqx
import jax
import jax.numpy as jnp
import joblib
import numpy as np
from jaxtyping import Array, Complex, Float, Int, Real

from qdots_qll.models.single_dot_weak_coupling_GAME import (
    clean_probabilities,
    true_parameters,
)
from data import Data
from experiments import ExperimentSingleDotWeakCouplingGAME

scaler = joblib.load("qdots_qll/models/scaler.job")

SEED = 5678

key = jax.random.key(SEED)
key, subkey = jax.random.split(key)

num_output_values = 12
n_times = 5000
n_params = 3000
time_dim = 1
param_dim = 4

n_cols = param_dim + time_dim + num_output_values
n_rows = n_params * n_times

in_size = param_dim + time_dim
out_size = num_output_values

# model = eqx.nn.MLP(
#     in_size=in_size,
#     out_size=out_size,
#     width_size=512,
#     depth=8,
#     final_activation=jax.nn.sigmoid,
#     key=subkey,
# )
# model = eqx.tree_deserialise_leaves("qdots_qll/models/temp_model.eqx", model)

model = eqx.nn.MLP(
    in_size=in_size,
    out_size=out_size,
    width_size=512,
    depth=10,
    final_activation=jax.nn.sigmoid,
    key=subkey,
)
model = eqx.tree_deserialise_leaves("qdots_qll/models/1e-7_512_10.eqx", model)

call_f_compiled = model.__call__


@eqx.filter_jit
def lkl_wrapped(particle, t):
    input = jnp.array([*particle, t])

    input = (input - scaler.mean_) / scaler.scale_

    output = call_f_compiled(input)

    output = jnp.array(
        [
            output.reshape(
                4,
                3,
            ),
            1
            - output.reshape(
                4,
                3,
            ),
        ]
    ).transpose(1, 2, 0)

    return output


class MLPSingleDotGAME(eqx.Module):
    true_parameters: Array

    def __init__(self, true_parameters=true_parameters):
        self.true_parameters = true_parameters

    def likelihood_particle(self, particle, t):
        return lkl_wrapped(particle, t)

    def likelihood_particle_with_basis_initial_state(
            self, particle, t, dist_initial_state, dist_measurement_basis
    ):
        lkl = self.likelihood_particle(particle, t)
        lkl = (
                dist_initial_state[:, None, None]
                * dist_measurement_basis[None, :, None]
                * lkl
        )
        return clean_probabilities(lkl)

    def fim(
            self,
            particle,
            t,
            prob_initial_state,
            prob_measurement_basis,
    ):
        lkl_outcome_array = self.likelihood_particle(particle, t)
        jac = jax.jacobian(self.likelihood_particle, argnums=0)(particle, t)
        jac = jac.reshape(
            jac.shape[0], -1
        )  # now we have flattened with respect to the basis and initial states

        p_i_p_j_over_lkloutcome = (
                1
                / lkl_outcome_array
                * prob_measurement_basis[None, :, None]
                * prob_initial_state[:, None, None]
        ).flatten()
        fim_element = jax.vmap(lambda x, p: jnp.outer(x, x) * p)(
            jac.T, p_i_p_j_over_lkloutcome
        )
        # return fim_element.sum(axis=0)
        return jnp.where(~jnp.isinf(fim_element), fim_element, 0).sum(axis=0)

    def generate_data(
            self, key, true_particle, t, initial_state_index, measurement_basis_index
    ):
        probabilities = self.likelihood_particle(true_particle, t)
        # probabilities has the shape [init rho, basis, prob_of_each_outcome]
        probability_given_state_and_basis = probabilities[
            initial_state_index, measurement_basis_index
        ]
        probability_given_state_and_basis = (
                probability_given_state_and_basis / probability_given_state_and_basis.sum()
        )

        no_outcomes = 2
        outcome = jax.random.choice(
            key, a=jnp.arange(no_outcomes), p=probability_given_state_and_basis
        )
        return jnp.array([initial_state_index, measurement_basis_index, outcome])

    def lkl_outcome_one_experiment(
            self, particle, experiment: ExperimentSingleDotWeakCouplingGAME
    ):
        time = jnp.squeeze(experiment.time)
        init_state = jnp.squeeze(experiment.initial_state)
        basis = jnp.squeeze(experiment.measurement_basis)
        return self.likelihood_particle(particle, time)[
            (init_state.astype(int)), (basis.astype(int))
        ]

    def measure_one_experiment(
            self, subkey, experiment: ExperimentSingleDotWeakCouplingGAME
    ):
        lkl = self.lkl_outcome_one_experiment(self.true_parameters, experiment)
        return jax.random.choice(subkey, jnp.array([0, 1]), p=lkl)

    # def measure_experiment(self, subkey, experiment: ExperimentSingleDotWeakCouplingGAME):

    def log_lkl_single_datum(self, particle, datum: Data):
        experiment = datum.experiment
        outcome = datum.outcome

        lkl = self.lkl_outcome_one_experiment(particle=particle, experiment=experiment)[
            outcome.astype(int)
        ]
        loglkl = jnp.log(lkl)
        return loglkl

    def total_log_lkl(self, particle, data: Data):
        def f_for_scan(carry, x):
            datum = x
            loglkl_datum = self.log_lkl_single_datum(particle, datum)
            carry = carry + loglkl_datum
            return carry, loglkl_datum

        sum_log_lkl, array_log_lkl_datum = jax.lax.scan(f_for_scan, init=0, xs=data)
        return sum_log_lkl - jnp.max(array_log_lkl_datum)

    def batch_total_log_lkl(self, particles, data: Data):
        return jax.vmap(self.total_log_lkl, in_axes=(0, None))(particles, data)

    def log_lkl_datum_multiple_particles(self, particles, datum: Data):
        loglkl_arr = jax.vmap(self.log_lkl_single_datum, in_axes=(0, None))(
            particles, datum
        )
        return jnp.squeeze(loglkl_arr)
