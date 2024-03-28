import qutip as qt
import numpy as np

sigmas_povm = (
    np.array(
        [
            [qt.ket2dm(a).full() for a in j.eigenstates()[1]]
            for j in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        ]
    ).reshape(-1, 2, 2)
    / 3
)
