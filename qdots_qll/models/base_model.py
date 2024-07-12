from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array

from qdots_qll.models.single_dot_weak_coupling_GAME import Data


class Model(eqx.Module):
    @abstractmethod
    def batch_total_log_lkl(particles, data: Data) -> Array:
        """Receives an array of n particles of dim m -> (n, m)
        and Data. Must return an array of shape (n, ) with the log_lkl
        of each particle.

        Args:
            particles (_type_): _description_
            data (Data): _description_

        Returns:
            Array: _description_
        """
        pass
