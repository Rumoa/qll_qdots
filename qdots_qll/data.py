import equinox as eqx
from jax import numpy as jnp

from qdots_qll.experiments import Experiment
from qdots_qll.utils.utils import ensure_array


class Data(eqx.Module):
    experiment: Experiment
    outcome: int

    def __init__(self, experiment: Experiment, outcome) -> None:
        self.experiment = experiment
        self.outcome = jnp.int8(ensure_array(outcome))

    def __iter__(self):
        for experiment, outcome in zip(self.experiment, self.outcome):
            yield Data(experiment, outcome)

    def __getitem__(self, item):
        return Data(self.experiment[item], self.outcome[item])

    def __len__(self):
        return len(self.outcome)

    def __str__(self):
        s = f"Experiment: {self.experiment}\noutcome: {self.outcome}"
        return s

    def append(self, other):
        updated_experiments = self.experiment.append(other.experiment)
        updated_outcomes = jnp.append(self.outcome, other.outcome)
        return Data(experiment=updated_experiments, outcome=updated_outcomes)
