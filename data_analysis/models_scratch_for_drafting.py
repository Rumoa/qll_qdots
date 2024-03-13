import jax
import jax.numpy as jnp
from qdots_qll.models.game import *

import qutip as qt

import matplotlib.pyplot as plt

from qbism import sic_povm

import jax.typing

import equinox as eqx


from jax import Array

# from jax.typing import ArrayLike
from jaxtyping import Array, Float, Complex, Int

seed = 3
rho_ex = qt.rand_dm_ginibre(2, seed=seed)


class base_class_dimension(eqx.Module):
    d: int
    matrix_d: Float[Array, "d d"]
    positive_eps: Float

    def __init__(
        self,
        dimension: int,
    ) -> None:
        self.d = dimension
        self.matrix_d = jnp.ones([self.d, self.d])
        self.positive_eps = 1e-8

    @jit
    def vec(self, rho: Complex[Array, "d d"]) -> Complex[Array, "d**2"]:
        return rho.flatten()

    @jit
    def sprepost(
        self, A: Complex[Array, "d1 d2"], B: Complex[Array, "d3 d4"]
    ) -> Float[Array, "d1*d3 d2*d4"]:
        return jnp.kron(A, B.T)

    @jit
    def spre(self, A: Complex[Array, "d d"]) -> Complex[Array, "d**2 d**2"]:
        d = A.shape[0]
        return self.sprepost(A, jnp.identity(d))

    @jit
    def spost(self, A: Complex[Array, "d d"]) -> Complex[Array, "d**2 d**2"]:
        d = A.shape[0]

        return self.sprepost(jnp.identity(d), A)


check_nan = jax.jit(
    lambda a: jax.lax.cond(jnp.isnan(a), lambda a: 0.0, lambda a: a, a)
)


@jit
def compute_P_superop(
    evolved_state_vec: Complex[Array, "d**2"],
    POVM_element_vec: Complex[Array, "d**2"],
):

    return check_nan(
        jnp.real(jnp.dot(dag(POVM_element_vec), evolved_state_vec))
    )


@jit
def compute_P_matrix_ver(
    evolved_state: Complex[Array, "d d"],
    POVM_element: Complex[Array, "d d"],
):

    return check_nan(jnp.real(jnp.trace(evolved_state @ POVM_element)))


class single_qdot(base_class_dimension):
    number_of_parameters: int
    delta: float
    Omega: float
    system_hamiltonian: Complex[Array, "d d"]
    A: Complex[Array, "d d"]
    POVM_arr: Complex[Array, "no_outcomes d d"]
    basis_elements: jax.Array
    # true_parameters: jax.Array

    def __init__(self, POVM_array: Complex[Array, "no_outcomes d d"]):
        super().__init__(dimension=2)
        self.number_of_parameters = 4
        self.delta = 0.12739334807998307
        self.Omega = 0.5
        self.system_hamiltonian = self.make_system_hamiltonian()
        self.A = jnp.array([[1, 0], [0, 0]])
        self.POVM_arr = POVM_array
        self.basis_elements = jnp.identity(4)

    @jit
    def make_system_hamiltonian(self) -> Complex[Array, "d d"]:
        return jnp.array([[self.delta, self.Omega / 2], [self.Omega / 2, 0]])

    @jit
    def make_liouvillian(
        self, particle: Float[Array, "number_of_parameters"]
    ) -> Complex[Array, "d**2 d**2"]:
        gn, gp, Sn, Sp = particle
        Snot = -self.delta
        gnot = (
            self.positive_eps
        )  # This is approx to zero. if we write zero derivatives explodes
        # because this appears in square roots d/dx (sqrt) prop to 1/sqrt

        U = jnp.linalg.eigh(self.system_hamiltonian)[1]
        Aij = U @ self.A @ dag(U)
        Cp = 0.5 * gp + 1j * Sp
        Cn = 0.5 * gn + 1j * Sn
        Cnot = 0.5 * gnot + 1j * Snot

        Gamma = jnp.array([[Cnot, Cn], [Cp, Cnot]])

        sqrtgamma = jnp.sqrt(jnp.real(Gamma))

        L = jnp.multiply(Aij, sqrtgamma)

        Af = jnp.multiply(Aij, jnp.conjugate(Gamma))
        Hrenorm = -1j / 2 * (Aij @ dag(Af) - Af @ dag(Aij))

        Htotal = U @ self.system_hamiltonian @ dag(U) + Hrenorm
        Liouvillian_ebasis = (
            -1j * (self.spre(Htotal) - self.spost(Htotal))
            + self.sprepost(dag(L), L)
            - 0.5 * (self.spre(L @ dag(L)) + self.spost(L @ dag(L)))
        )
        return (
            self.sprepost(U, dag(U))
            @ Liouvillian_ebasis
            @ dag(self.sprepost(U, dag(U)))
        )

    @jit
    def evolve_initial_state_liouvillian(
        self,
        t: Float,
        liouvillian: Complex[Array, "d**2 d**2"],
        initial_state: Complex[Array, "d**2"],
    ) -> Complex[Array, "d**2"]:
        return expm(t * liouvillian) @ initial_state

    @jit
    def likelihood_particle(
        self,
        particle: Float[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Float[Array, "no_outcomes"]:
        liouvillian = self.make_liouvillian(particle)
        evolved_state_vec = self.evolve_initial_state_liouvillian(
            t, liouvillian, initial_state
        )
        evolved_state_mat = (evolved_state_vec).reshape(self.matrix_d.shape)
        probability = jax.vmap(compute_P_matrix_ver, in_axes=(None, 0))(
            evolved_state_mat, self.POVM_arr
        )
        return probability

    @jit
    def fim(
        self,
        particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Float[Array, "number_of_parameters number_of_parameters"]:
        p_array = self.likelihood_particle(particle, t, initial_state)

        jacobian = jax.jacobian(self.likelihood_particle, 0)(
            particle, t, initial_state
        )

        example_zero_matrix = jnp.zeros([particle.shape[0], particle.shape[0]])

        fim_element = jax.vmap(lambda x, p: jnp.outer(x, x) / p)(
            jacobian, p_array
        )
        return jnp.where(~jnp.isinf(fim_element), fim_element, 0).sum(axis=0)

    @jit
    def generate_data(
        self,
        key: Int[Array, "2"],
        true_particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Int[Array, "1"]:
        probabilities = self.likelihood_particle(
            true_particle, t, initial_state
        )
        probabilities = probabilities / probabilities.sum()
        no_of_outcomes = self.POVM_arr.shape[0]
        outcome = jax.random.choice(
            key, a=jnp.arange(no_of_outcomes), p=probabilities
        )
        return outcome

    @jit
    def qfim(
        self,
        particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Float[Array, "number_of_parameters number_of_parameters"]:

        d = self.matrix_d.shape[0]
        no_parameters = particle.shape[0]

        L = self.make_liouvillian(particle)
        evolved_state = self.evolve_initial_state_liouvillian(
            t, L, initial_state
        ).reshape(d, d)

        eigvals, eigvecs = jnp.linalg.eigh(evolved_state)

        partial_rhoi_parameterj = jax.jacobian(
            lambda particle: self.evolve_initial_state_liouvillian(
                t, self.make_liouvillian(particle), initial_state
            ),
            holomorphic=True,
        )(particle.astype(jnp.complex64)).T.reshape(
            no_parameters, d, d
        )  # first index is related to the parameter

        denominator_lambda_i_plus_lambda_j = jax.vmap(
            lambda i: jax.vmap(lambda j: 1 / (i + j))(eigvals)
        )(eigvals)

        braket = jnp.einsum(
            "im,amn,jn-> ija",
            jnp.conjugate(eigvecs),
            partial_rhoi_parameterj,
            eigvecs,
        )

        braket_with_denominator_product = jnp.einsum(
            "ija,  ij-> ija", braket, denominator_lambda_i_plus_lambda_j
        )

        braket_with_denominator_product_cleaned = jnp.where(
            ~jnp.isinf(braket_with_denominator_product),
            braket_with_denominator_product,
            jnp.zeros(braket_with_denominator_product.shape),
        )

        qfim = 2 * jnp.real(
            jnp.einsum(
                "ija, ijb-> ab",
                jnp.conjugate(braket),
                braket_with_denominator_product_cleaned,
            )
        )
        return qfim


check_nan = jax.jit(
    lambda a: jax.lax.cond(jnp.isnan(a), lambda a: 0.0, lambda a: a, a)
)


@jit
def compute_P_superop(
    evolved_state_vec: Complex[Array, "d**2"],
    POVM_element_vec: Complex[Array, "d**2"],
):

    return check_nan(
        jnp.real(jnp.dot(dag(POVM_element_vec), evolved_state_vec))
    )


@jit
def compute_P_matrix_ver(
    evolved_state: Complex[Array, "d d"],
    POVM_element: Complex[Array, "d d"],
):

    return check_nan(jnp.real(jnp.trace(evolved_state @ POVM_element)))


class two_qdots_separable_maps(base_class_dimension):
    number_of_parameters: int
    delta: float
    Omega: float
    one_dot_system_hamiltonian: Complex[Array, "2 2"]
    A: Complex[Array, "2 2"]
    POVM_arr: Complex[Array, "no_outcomes 4 4"]
    basis_elements: jax.Array
    # true_parameters: jax.Array

    def __init__(self, POVM_array: Complex[Array, "no_outcomes 4 4"]):
        super().__init__(dimension=4)
        self.number_of_parameters = 4
        self.delta = 0.12739334807998307
        self.Omega = 0.5
        self.one_dot_system_hamiltonian = (
            self.make_one_dot_system_hamiltonian()
        )
        self.A = jnp.array([[1, 0], [0, 0]])
        self.POVM_arr = POVM_array
        self.basis_elements = jnp.identity(4)

    @jit
    def make_one_dot_system_hamiltonian(self) -> Complex[Array, "2 2"]:
        return jnp.array([[self.delta, self.Omega / 2], [self.Omega / 2, 0]])

    @jit
    def make_one_dot_liouvillian(
        self, particle: Float[Array, "number_of_parameters"]
    ) -> Complex[Array, "2**2 2**2"]:
        gn, gp, Sn, Sp = particle
        Snot = -self.delta
        gnot = (
            self.positive_eps
        )  # This is approx to zero. if we write zero derivatives explodes
        # because this appears in square roots d/dx (sqrt) prop to 1/sqrt

        U = jnp.linalg.eigh(self.one_dot_system_hamiltonian)[1]
        Aij = U @ self.A @ dag(U)
        Cp = 0.5 * gp + 1j * Sp
        Cn = 0.5 * gn + 1j * Sn
        Cnot = 0.5 * gnot + 1j * Snot

        Gamma = jnp.array([[Cnot, Cn], [Cp, Cnot]])

        sqrtgamma = jnp.sqrt(jnp.real(Gamma))

        L = jnp.multiply(Aij, sqrtgamma)

        Af = jnp.multiply(Aij, jnp.conjugate(Gamma))
        Hrenorm = -1j / 2 * (Aij @ dag(Af) - Af @ dag(Aij))

        Htotal = U @ self.one_dot_system_hamiltonian @ dag(U) + Hrenorm
        Liouvillian_ebasis = (
            -1j * (self.spre(Htotal) - self.spost(Htotal))
            + self.sprepost(dag(L), L)
            - 0.5 * (self.spre(L @ dag(L)) + self.spost(L @ dag(L)))
        )
        return (
            self.sprepost(U, dag(U))
            @ Liouvillian_ebasis
            @ dag(self.sprepost(U, dag(U)))
        )

    @jit
    def evolve_initial_state_with_one_dot_liouvillian(
        self,
        t: Float,
        liouvillian: Complex[Array, "2**2 2**2"],
        initial_state: Complex[Array, "4**2"],
    ) -> Complex[Array, "4**2"]:

        # This is the tricky part, we take the liouvillian, and then
        # we will apply both exp(L) to each qubit, carefully.
        # Remember the liouvillian is only for one dot

        map_A = expm(t * liouvillian)
        map_B = map_A

        total_map_superop = jnp.kron(
            map_A, map_B
        )  # this now is a superop in 16x16

        # Now the initial state needs to be reshape to 2, 2, 2, 2
        # swap the 2nd and 3d index.
        # vectorize
        # apply the total_map_superop
        # convert to 2, 2, 2, 2
        # unswap 2nd and 3rd
        # vectorize
        evolved_state = vec(
            (
                total_map_superop
                @ vec(
                    initial_state.reshape([2, 2, 2, 2])
                    .swapaxes(1, 2)
                    .reshape([4, 4])
                )
            )
            .reshape([2, 2, 2, 2])
            .swapaxes(1, 2)
            .reshape([4, 4])
        )

        return evolved_state

    # @jit
    # def evolve_initial_state_with_one_dot_liouvillian(
    #     self,
    #     t: Float,
    #     liouvillian: Complex[Array, "2**2 2**2"],
    #     initial_state: Complex[Array, "4**2"],
    # ) -> Complex[Array, "4**2"]:

    #     # This is the tricky part, we take the liouvillian, and then
    #     # we will apply both exp(L) to each qubit, carefully.
    #     # Remember the liouvillian is only for one dot

    #     map_A = expm(t * liouvillian)
    #     map_B = map_A

    #     total_map_superop = jnp.kron(
    #          jnp.identity(4), map_A
    #     )  # this now is a superop in 16x16

    #     # Now the initial state needs to be reshape to 2, 2, 2, 2
    #     # swap the 2nd and 3d index.
    #     # vectorize
    #     # apply the total_map_superop
    #     # convert to 2, 2, 2, 2
    #     # unswap 2nd and 3rd
    #     # vectorize
    #     evolved_state = vec(
    #         (
    #             total_map_superop
    #             @ vec(initial_state.reshape([2, 2, 2, 2]).swapaxes(1, 2).reshape([4, 4]))
    #         )
    #         .reshape([2, 2, 2, 2])
    #         .swapaxes(1, 2)
    #         .reshape([4, 4])
    #     )

    #     return evolved_state

    @jit
    def likelihood_particle(
        self,
        particle: Float[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "4**2"],
    ) -> Float[Array, "no_outcomes"]:
        liouvillian = self.make_one_dot_liouvillian(particle)
        evolved_state_vec = self.evolve_initial_state_with_one_dot_liouvillian(
            t, liouvillian, initial_state
        )
        evolved_state_mat = (evolved_state_vec).reshape(self.matrix_d.shape)
        probability = jax.vmap(compute_P_matrix_ver, in_axes=(None, 0))(
            evolved_state_mat, self.POVM_arr
        )
        return probability

    @jit
    def fim(
        self,
        particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Float[Array, "number_of_parameters number_of_parameters"]:
        p_array = self.likelihood_particle(particle, t, initial_state)

        jacobian = jax.jacobian(self.likelihood_particle, 0)(
            particle, t, initial_state
        )

        # example_zero_matrix = jnp.zeros([particle.shape[0], particle.shape[0]])

        fim_element = jax.vmap(lambda x, p: jnp.outer(x, x) / p)(
            jacobian, p_array
        )
        return jnp.where(~jnp.isinf(fim_element), fim_element, 0).sum(axis=0)

    @jit
    def generate_data(
        self,
        key: Int[Array, "2"],
        true_particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Int[Array, "1"]:
        probabilities = self.likelihood_particle(
            true_particle, t, initial_state
        )
        probabilities = probabilities / probabilities.sum()
        no_of_outcomes = self.POVM_arr.shape[0]
        outcome = jax.random.choice(
            key, a=jnp.arange(no_of_outcomes), p=probabilities
        )
        return outcome

    @jit
    def qfim(
        self,
        particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Float[Array, "number_of_parameters number_of_parameters"]:

        d = self.matrix_d.shape[0]
        no_parameters = particle.shape[0]

        L = self.make_one_dot_liouvillian(particle)
        evolved_state = self.evolve_initial_state_with_one_dot_liouvillian(
            t, L, initial_state
        ).reshape(d, d)

        eigvals, eigvecs = jnp.linalg.eigh(evolved_state)

        partial_rhoi_parameterj = jax.jacobian(
            lambda particle: self.evolve_initial_state_with_one_dot_liouvillian(
                t, self.make_one_dot_liouvillian(particle), initial_state
            ),
            holomorphic=True,
        )(particle.astype(jnp.complex64)).T.reshape(
            no_parameters, d, d
        )  # first index is related to the parameter

        denominator_lambda_i_plus_lambda_j = jax.vmap(
            lambda i: jax.vmap(lambda j: 1 / (i + j))(eigvals)
        )(eigvals)

        braket = jnp.einsum(
            "im,amn,jn-> ija",
            jnp.conjugate(eigvecs),
            partial_rhoi_parameterj,
            eigvecs,
        )

        braket_with_denominator_product = jnp.einsum(
            "ija,  ij-> ija", braket, denominator_lambda_i_plus_lambda_j
        )

        braket_with_denominator_product_cleaned = jnp.where(
            ~jnp.isinf(braket_with_denominator_product),
            braket_with_denominator_product,
            jnp.zeros(braket_with_denominator_product.shape),
        )

        qfim = 2 * jnp.real(
            jnp.einsum(
                "ija, ijb-> ab",
                jnp.conjugate(braket),
                braket_with_denominator_product_cleaned,
            )
        )
        return qfim

    @jit
    def qfim_with_inverse(
        self,
        particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Float[Array, "number_of_parameters number_of_parameters"]:

        d = self.matrix_d.shape[0]
        no_parameters = particle.shape[0]

        L = self.make_one_dot_liouvillian(particle)
        evolved_state = self.evolve_initial_state_with_one_dot_liouvillian(
            t, L, initial_state
        ).reshape(d, d)

        middle_product = jnp.linalg.inv(
            jnp.kron(evolved_state, jnp.identity(d))
            + jnp.kron(
                jnp.identity(d),
                jnp.conjugate(evolved_state),
            )
        )

        partial_rhoi_parameterj = jax.jacobian(
            lambda particle: self.evolve_initial_state_with_one_dot_liouvillian(
                t, self.make_one_dot_liouvillian(particle), initial_state
            ),
            holomorphic=True,
        )(particle.astype(jnp.complex64)).T.reshape(
            no_parameters, -1
        )  # first index is related to the parameter

        qfim = 2 * jnp.einsum(
            "iv, vw, jw",
            jnp.conjugate(partial_rhoi_parameterj),
            middle_product,
            partial_rhoi_parameterj,
        )

        return qfim

    @jit
    def make_SLD(
        self,
        particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Float[Array, "number_of_parameters number_of_parameters"]:

        d = self.matrix_d.shape[0]
        no_parameters = particle.shape[0]

        L = self.make_one_dot_liouvillian(particle)
        evolved_state = self.evolve_initial_state_with_one_dot_liouvillian(
            t, L, initial_state
        ).reshape(d, d)

        # eigvals, eigvecs = jnp.linalg.eigh(evolved_state)

        partial_rhoi_parameterj = jax.jacobian(
            lambda particle: self.evolve_initial_state_with_one_dot_liouvillian(
                t, self.make_one_dot_liouvillian(particle), initial_state
            ),
            holomorphic=True,
        )(
            particle.astype(jnp.complex64)
        ).T  # first index is related to the parameter

        sld = jnp.einsum(
            "jk, ik -> ij",
            2
            * jnp.linalg.inv(
                jnp.kron(evolved_state, jnp.identity(d))
                + jnp.kron(
                    jnp.identity(d),
                    evolved_state,
                )
            ),
            partial_rhoi_parameterj,
        )

        return sld


class two_qdots_identity_for_systemB(base_class_dimension):
    number_of_parameters: int
    delta: float
    Omega: float
    one_dot_system_hamiltonian: Complex[Array, "2 2"]
    A: Complex[Array, "2 2"]
    POVM_arr: Complex[Array, "no_outcomes 4 4"]
    basis_elements: jax.Array
    # true_parameters: jax.Array

    def __init__(self, POVM_array: Complex[Array, "no_outcomes 4 4"]):
        super().__init__(dimension=4)
        self.number_of_parameters = 4
        self.delta = 0.12739334807998307
        self.Omega = 0.5
        self.one_dot_system_hamiltonian = (
            self.make_one_dot_system_hamiltonian()
        )
        self.A = jnp.array([[1, 0], [0, 0]])
        self.POVM_arr = POVM_array
        self.basis_elements = jnp.identity(4)

    @jit
    def make_one_dot_system_hamiltonian(self) -> Complex[Array, "2 2"]:
        return jnp.array([[self.delta, self.Omega / 2], [self.Omega / 2, 0]])

    @jit
    def make_one_dot_liouvillian(
        self, particle: Float[Array, "number_of_parameters"]
    ) -> Complex[Array, "2**2 2**2"]:
        gn, gp, Sn, Sp = particle
        Snot = -self.delta
        gnot = (
            self.positive_eps
        )  # This is approx to zero. if we write zero derivatives explodes
        # because this appears in square roots d/dx (sqrt) prop to 1/sqrt

        U = jnp.linalg.eigh(self.one_dot_system_hamiltonian)[1]
        Aij = U @ self.A @ dag(U)
        Cp = 0.5 * gp + 1j * Sp
        Cn = 0.5 * gn + 1j * Sn
        Cnot = 0.5 * gnot + 1j * Snot

        Gamma = jnp.array([[Cnot, Cn], [Cp, Cnot]])

        sqrtgamma = jnp.sqrt(jnp.real(Gamma))

        L = jnp.multiply(Aij, sqrtgamma)

        Af = jnp.multiply(Aij, jnp.conjugate(Gamma))
        Hrenorm = -1j / 2 * (Aij @ dag(Af) - Af @ dag(Aij))

        Htotal = U @ self.one_dot_system_hamiltonian @ dag(U) + Hrenorm
        Liouvillian_ebasis = (
            -1j * (self.spre(Htotal) - self.spost(Htotal))
            + self.sprepost(dag(L), L)
            - 0.5 * (self.spre(L @ dag(L)) + self.spost(L @ dag(L)))
        )
        return (
            self.sprepost(U, dag(U))
            @ Liouvillian_ebasis
            @ dag(self.sprepost(U, dag(U)))
        )

    @jit
    def evolve_initial_state_with_one_dot_liouvillian(
        self,
        t: Float,
        liouvillian: Complex[Array, "2**2 2**2"],
        initial_state: Complex[Array, "4**2"],
    ) -> Complex[Array, "4**2"]:

        # This is the tricky part, we take the liouvillian, and then
        # we will apply both exp(L) to each qubit, carefully.
        # Remember the liouvillian is only for one dot

        map_A = expm(t * liouvillian)

        total_map_superop = jnp.kron(
            map_A, jnp.identity(4)
        )  # this now is a superop in 16x16

        # Now the initial state needs to be reshape to 2, 2, 2, 2
        # swap the 2nd and 3d index.
        # vectorize
        # apply the total_map_superop
        # convert to 2, 2, 2, 2
        # unswap 2nd and 3rd
        # vectorize
        evolved_state = vec(
            (
                total_map_superop
                @ vec(
                    initial_state.reshape([2, 2, 2, 2])
                    .swapaxes(1, 2)
                    .reshape([4, 4])
                )
            )
            .reshape([2, 2, 2, 2])
            .swapaxes(1, 2)
            .reshape([4, 4])
        )

        return evolved_state

    @jit
    def likelihood_particle(
        self,
        particle: Float[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "4**2"],
    ) -> Float[Array, "no_outcomes"]:
        liouvillian = self.make_one_dot_liouvillian(particle)
        evolved_state_vec = self.evolve_initial_state_with_one_dot_liouvillian(
            t, liouvillian, initial_state
        )
        evolved_state_mat = (evolved_state_vec).reshape(self.matrix_d.shape)
        probability = jax.vmap(compute_P_matrix_ver, in_axes=(None, 0))(
            evolved_state_mat, self.POVM_arr
        )
        return probability

    @jit
    def fim(
        self,
        particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Float[Array, "number_of_parameters number_of_parameters"]:
        p_array = self.likelihood_particle(particle, t, initial_state)

        jacobian = jax.jacobian(self.likelihood_particle, 0)(
            particle, t, initial_state
        )

        example_zero_matrix = jnp.zeros([particle.shape[0], particle.shape[0]])

        fim_element = jax.vmap(lambda x, p: jnp.outer(x, x) / p)(
            jacobian, p_array
        )
        return jnp.where(~jnp.isinf(fim_element), fim_element, 0).sum(axis=0)

    @jit
    def generate_data(
        self,
        key: Int[Array, "2"],
        true_particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Int[Array, "1"]:
        probabilities = self.likelihood_particle(
            true_particle, t, initial_state
        )
        probabilities = probabilities / probabilities.sum()
        no_of_outcomes = self.POVM_arr.shape[0]
        outcome = jax.random.choice(
            key, a=jnp.arange(no_of_outcomes), p=probabilities
        )
        return outcome

    @jit
    def qfim(
        self,
        particle: Complex[Array, "number_of_parameters"],
        t: Float,
        initial_state: Complex[Array, "d**2"],
    ) -> Float[Array, "number_of_parameters number_of_parameters"]:

        d = self.matrix_d.shape[0]
        no_parameters = particle.shape[0]

        L = self.make_one_dot_liouvillian(particle)
        evolved_state = self.evolve_initial_state_with_one_dot_liouvillian(
            t, L, initial_state
        ).reshape(d, d)

        eigvals, eigvecs = jnp.linalg.eigh(evolved_state)

        partial_rhoi_parameterj = jax.jacobian(
            lambda particle: self.evolve_initial_state_with_one_dot_liouvillian(
                t, self.make_one_dot_liouvillian(particle), initial_state
            ),
            holomorphic=True,
        )(particle.astype(jnp.complex64)).T.reshape(
            no_parameters, d, d
        )  # first index is related to the parameter

        denominator_lambda_i_plus_lambda_j = jax.vmap(
            lambda i: jax.vmap(lambda j: 1 / (i + j))(eigvals)
        )(eigvals)

        braket = jnp.einsum(
            "im,amn,jn-> ija",
            jnp.conjugate(eigvecs),
            partial_rhoi_parameterj,
            eigvecs,
        )

        braket_with_denominator_product = jnp.einsum(
            "ija,  ij-> ija", braket, denominator_lambda_i_plus_lambda_j
        )

        braket_with_denominator_product_cleaned = jnp.where(
            ~jnp.isinf(braket_with_denominator_product),
            braket_with_denominator_product,
            jnp.zeros(braket_with_denominator_product.shape),
        )

        qfim = 2 * jnp.real(
            jnp.einsum(
                "ija, ijb-> ab",
                jnp.conjugate(braket),
                braket_with_denominator_product_cleaned,
            )
        )
        return qfim
