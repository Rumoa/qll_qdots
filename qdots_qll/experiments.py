import equinox as eqx
import jax
from jax import numpy as jnp

from qdots_qll.utils.utils import ensure_array


class Experiment(eqx.Module):
    pass


class ExperimentSingleDotWeakCouplingGAME(Experiment):
    time: float
    initial_state: int
    measurement_basis: int

    def __init__(self, t: float, initial_state: int, measurement_basis: int) -> None:
        self.time = jnp.float32(ensure_array(t))
        self.initial_state = jnp.int8(ensure_array(initial_state))
        self.measurement_basis = jnp.int8(ensure_array(measurement_basis))

    def __len__(self):
        return len(self.time)

    def __iter__(self):
        for i in range(len(self)):
            yield ExperimentSingleDotWeakCouplingGAME(
                self.time[i], self.initial_state[i], self.measurement_basis[i]
            )
        # for t in self.time:
        #     yield Experiment(t)

    def __getitem__(self, item):
        return ExperimentSingleDotWeakCouplingGAME(
            self.time[item], self.initial_state[item], self.measurement_basis[item]
        )

    def __str__(self):
        s = f"Time {self.time}\nInitial state {self.initial_state}\nMeasurement basis {self.measurement_basis} "
        return s

    def append(
        self, other: "ExperimentSingleDotWeakCouplingGAME"
    ) -> "ExperimentSingleDotWeakCouplingGAME":
        def append_if_array(x, y):
            if isinstance(x, jnp.ndarray):
                return jnp.append(x, y)
            return x  # If it's not an array, keep it unchanged

        new_fields = jax.tree_util.tree_map(append_if_array, self, other)
        return ExperimentSingleDotWeakCouplingGAME(
            new_fields.time, new_fields.initial_state, new_fields.measurement_basis
        )
