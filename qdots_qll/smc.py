# Here we write the functions associated to the smc update.
# TODO: Make things cleaner bc it looks a bit sloppy


import equinox as eqx
import jax
from jax import jit
from jaxtyping import Array

from qdots_qll.distributions import Distribution, update_log_weights
from qdots_qll.exp_design import ExperimentalDesign
from qdots_qll.data import Data
from qdots_qll.resamplers import Resampler


@jax.jit
def replace_single_exp(all_exps, new_exp, position):
    """_summary_"""
    arrs, structure = jax.tree.flatten(all_exps)
    arr_e, _ = jax.tree.flatten(new_exp)
    for j in range(len(arrs)):
        arrs[j] = arrs[j].at[position].set(arr_e[j])
    new_empty = jax.tree.unflatten(structure, arrs)
    return new_empty


@jax.jit
def replace_single_datum(all_data, new_datum, position):
    structure_all_data = eqx.tree_flatten_one_level(all_data)[1]

    all_exps = all_data.experiment
    all_outcomes = all_data.outcome

    exp_to_be_placed = new_datum.experiment
    outcome_to_be_placed = new_datum.outcome

    new_outcomes = all_outcomes.at[position].set(outcome_to_be_placed)

    replaced_exps = replace_single_exp(all_exps, exp_to_be_placed, position)

    return jax.tree.unflatten(structure_all_data, [replaced_exps, new_outcomes])


class SMCUpdater(eqx.Module):
    model: eqx.Module
    exp_design: ExperimentalDesign
    resampler: Resampler

    def __init__(self, model, exp_design, resampler) -> None:
        self.model = model
        self.exp_design = exp_design
        self.resampler = resampler

    @jit
    def step(
        self,
        key: Array,
        iteration: int,
        distribution: Distribution,
        data: Data,
        *args,
        **kwargs,
    ) -> tuple[Array, Distribution]:
        # generate experiment
        # Measure experiment
        # create datum
        # append to data
        # update distribution
        # resample if necessary
        key, subkey = jax.random.split(key)
        # experiment = e1
        experiment = self.exp_design.generate_experiment(
            model=self.model,
            distribution=distribution,
            data=data,
            subkey=subkey,
        )

        outcome = self.model.measure_one_experiment(subkey, experiment)
        datum = Data(experiment, outcome)

        log_lkl = self.model.log_lkl_datum_multiple_particles(
            distribution.particles_locations, datum
        )

        data = replace_single_datum(data, datum, iteration)

        distribution: Distribution = update_log_weights(
            dist=distribution, new_log_lkl=log_lkl
        )

        key, subkey = jax.random.split(key)
        key, distribution = jax.lax.cond(
            distribution.check_resampling(),
            _resample,
            _do_not_resample,
            *(key, distribution, iteration, data, self.resampler),
        )

        iteration = iteration + 1
        return key, iteration, distribution, data


def _do_not_resample(key, dist: Distribution, *args, **kwargs):
    return key, dist


def _resample(key, dist, iteration, data, resampler, *args, **kwargs):
    key, subkey = jax.random.split(key)
    new_dist: Distribution = resampler.resample(
        subkey=subkey,
        index_data=iteration,
        distribution=dist,
        data=data,
    )
    return key, new_dist
