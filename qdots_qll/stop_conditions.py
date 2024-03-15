import jax
from jax import jit
import equinox as eqx


class TerminationChecker(eqx.Module):
    max_iterations: float

    def __init__(self, max_iterations):
        self.max_iterations = max_iterations

    @jit
    def check_stop(self, run):
        return self._iter_lower_than_max_iter(run)

    @jit
    def _iter_lower_than_max_iter(self, run):
        iter = run.iteration
        return jax.lax.cond(
            iter < self.max_iterations, lambda _: True, lambda _: False, None
        )
