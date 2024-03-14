import jax
from jax import jit
import equinox as eqx


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
