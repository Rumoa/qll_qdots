import jax.numpy as jnp
from itertools import product
import qutip as qt


def vec(a):
    return a.flatten()


names = [
    "$\\gamma ( - \\eta)$",
    "$\\gamma ( + \\eta)$",
    "$S ( - \\eta)$",
    "$S ( +\\eta)$",
]


names_outcomes = [
    "+x",
    "-x",
    "+y",
    "-y",
    "+z",
    "-z",
]
outcomes_string_list = list(product(names_outcomes, names_outcomes))

canonical_povm_qubit = (
    jnp.array(
        [
            [0.5 * (qt.identity(2) + mat), 0.5 * (qt.identity(2) - mat)]
            for mat in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        ]
    ).reshape(-1, 2, 2)
    / 3
)

povm_local = jnp.array(
    [
        jnp.kron(i[0], i[1])
        for i in product(canonical_povm_qubit, canonical_povm_qubit)
    ]
)

qubit_0_states = [qt.basis(2, 0), qt.basis(2, 1)]

max_entangled_dm_vec = vec(
    jnp.array(
        sum(
            [
                qt.tensor(q0 * q1.dag(), (q0 * q1.dag()))
                for q0 in qubit_0_states
                for q1 in qubit_0_states
            ]
        )
        .unit()
        .full()
    )
)
