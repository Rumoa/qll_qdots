import jax
import jax.numpy as jnp
import qutip as qt
from jax.scipy.linalg import expm
from jaxtyping import Array, Complex, Float

from qdots_qll.data import Data
from qdots_qll.experiments import ExperimentSingleDotWeakCouplingGAME
from qdots_qll.models.models_scratch_for_drafting import BaseClassDimension

# These parameters are related to the ones used in the paper:
# [1] A. Nazir and D. P. S. McCutcheon, Modelling Exciton-Phonon Interactions
# in Optically Driven Quantum Dots, J. Phys.: Condens. Matter 28, 103002 (2016).
# in the weak coupling regime (FIG. 1). Since we use the GAME master equation,
# we need to double the decay rates to match the same behaviour.
_G = jnp.array(
    [
        jnp.array(
            [[1, 0], [0, 1]],
        ),
        jnp.array(
            [[0, 1], [1, 0]],
        ),
        jnp.array(
            [[0, -1j], [1j, 0]],
        ),
        jnp.array(
            [[1, 0], [0, -1]],
        ),
    ]
) / jnp.sqrt(2)

is_probability_correct = lambda p: jnp.logical_and((p >= 0.0), (p <= 1.0))
trim_invalid_probs = lambda prob_array: jnp.where(
    is_probability_correct(prob_array), prob_array, jnp.abs(prob_array) * 0
)

trim_nan_probs = lambda prob_array: jnp.where(
    ~jnp.isnan(prob_array), prob_array, jnp.abs(prob_array) * 0
)


def clean_probabilities(prob_array):
    return trim_nan_probs(trim_invalid_probs(prob_array))


def rho_to_bloch(rho):
    return jnp.einsum("ijk,kj-> i", _G, rho).real


def bloch_to_rho(bloch_v):
    return jnp.einsum("jkl, j", _G, bloch_v)


gamma_minus = 0.15710846160566203
gamma_plus = 0.17916503425352892
S_minus = 0.053851494081252074
S_plus = -0.3336948226536299

true_parameters = jnp.array([2 * gamma_minus, 2 * gamma_plus, S_minus, S_plus])

canonical_povm = (
    jnp.array(
        [
            qt.identity(2).full() + qt.sigmax().full(),
            qt.identity(2).full() - qt.sigmax().full(),
            qt.identity(2).full() + qt.sigmay().full(),
            qt.identity(2).full() - qt.sigmay().full(),
            qt.identity(2).full() + qt.sigmaz().full(),
            qt.identity(2).full() - qt.sigmaz().full(),
        ]
    )
    / 2
).reshape(-1, 2, 2, 2)

zero = qt.basis(2, 0)
one = qt.basis(2, 1)
plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
minus = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()

initial_states = [zero, one, plus, minus]

initial_states_dm = jnp.array([qt.ket2dm(i).full() for i in initial_states])

initial_states_bloch = jax.vmap(rho_to_bloch)(initial_states_dm)


class SingleDotWeakCouplingGAME(BaseClassDimension):
    number_of_parameters: int
    delta: float
    Omega: float
    true_parameters: Array
    T: float
    POVM_arr: Complex[Array, "no_basis no_outcomes d d"]
    initial_states_bloch: Float[Array, "no_initial_states d"]
    basis_elements: jax.Array
    trace_povm_G: Float[Array, "no_outcomes d"]
    system_hamiltonian: Float[Array, "2 2"]
    Aij: Complex[Array, "2 2"]
    U: Complex[Array, "2 2"]
    matrix_change_basis_bloch: Complex[Array, "2 2"]
    vec_G: Complex[Array, "d d d"]

    def __init__(self, true_parameters=true_parameters):
        super().__init__(
            dimension=2,
        )
        self.number_of_parameters = 4
        self.delta = 0.12739334807998307
        self.Omega = 0.5
        self.T = 30
        self.true_parameters = true_parameters
        self.POVM_arr = canonical_povm
        self.basis_elements = jnp.identity(4)
        self.initial_states_bloch = initial_states_bloch
        self.trace_povm_G = jnp.einsum("ijkm,lmk", self.POVM_arr, _G).real
        self.system_hamiltonian = self.make_system_hamiltonian()
        self.Aij = self.make_Aij()[0]
        self.U = self.make_Aij()[1]
        self.matrix_change_basis_bloch = self.bloch_matrix_change_of_basis(self.U)

        self.vec_G = jax.vmap(lambda g: self.vec(g))(_G)

    def make_bloch_matrix(self, particle):
        gn, gp, Sn, Sp = particle
        gnot = 1e-9
        Snot = -self.delta
        # system_hamiltonian = self.delta * jnp.array([[1, 0], [0, -1]]) / 2 + self.Omega * jnp.array(
        #     [[0, 1], [1, 0]]) / 2
        # system_hamiltonian = self.make_system_hamiltonian()

        # A = jnp.array([[1, 0], [0, -1]])/2
        # Aij, U = self.make_Aij()
        Aij = self.Aij
        U = self.U

        Cp = 0.5 * gp + 1j * Sp
        Cn = 0.5 * gn + 1j * Sn
        Cnot = 0.5 * gnot + 1j * Snot
        Gamma = jnp.array([[Cnot, Cn], [Cp, Cnot]])

        sqrtgamma = jnp.sqrt(
            jnp.real(Gamma).astype("complex64")
        )  # This had a bug before. since we take the real part of Gamma, the array into jnp.sqrt is real and outputs nan if any of the elements is negative
        # solved with astype(complex)
        L = jnp.multiply(Aij, sqrtgamma)

        Af = jnp.multiply(Aij, jnp.conjugate(Gamma))

        H_renormalized = -1j / 2 * (Aij @ self.dag(Af) - Af @ self.dag(Aij))

        Htotal = U @ self.system_hamiltonian @ self.dag(U) + H_renormalized
        liouvillian_energy_basis = (
            -1j * (self.spre(Htotal) - self.spost(Htotal))
            + self.sprepost(self.dag(L), L)
            - 0.5 * (self.spre(L @ self.dag(L)) + self.spost(L @ self.dag(L)))
        )

        matrix_change_basis_bloch = self.matrix_change_basis_bloch

        vec_G = self.vec_G

        map_bloch_energy_basis = jnp.einsum(
            "ij,jk,lk-> il", jnp.conjugate(vec_G), liouvillian_energy_basis, vec_G
        )

        map_bloch = (
            matrix_change_basis_bloch
            @ map_bloch_energy_basis
            @ matrix_change_basis_bloch.T
        )
        return map_bloch

    def bloch_matrix_change_of_basis(self, U):
        matrix_change_basis_bloch = jnp.einsum(
            "kl,ilm,mn,jnk->ij", self.dag(U), _G, U, _G
        )
        return matrix_change_basis_bloch

    def make_Aij(self):
        A = jnp.array([[1, 0], [0, 0]])
        U = jnp.linalg.eigh(self.system_hamiltonian)[1]
        Aij = U @ A @ self.dag(U)
        return Aij, U

    def make_system_hamiltonian(self):
        system_hamiltonian = (
            self.delta * jnp.array([[1, 0], [0, 0]])
            + self.Omega * jnp.array([[0, 1], [1, 0]]) / 2
        )
        return system_hamiltonian

    def likelihood_particle(self, particle, t):
        M = self.make_bloch_matrix(particle)
        expMt = expm(M * t)
        evolved_vectors_states = jax.vmap(lambda v0: expMt @ v0)(
            self.initial_states_bloch
        )
        p_outcome = jnp.einsum(
            "iz,jkz-> ijk", evolved_vectors_states, self.trace_povm_G
        ).real
        # Notation: [init rho, basis, prob_of_each_outcome]
        return clean_probabilities(p_outcome)

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

    # def fim(
    #     self,
    #     particle,
    #     t,
    #     prob_initial_state,
    #     prob_measurement_basis,
    # ):
    #     prob_array = self.likelihood_particle_with_basis_initial_state(
    #         particle, t, prob_initial_state, prob_measurement_basis
    #     )
    #     jac = jax.jacobian(
    #         self.likelihood_particle_with_basis_initial_state, argnums=0
    #     )(particle, t, prob_initial_state, prob_measurement_basis)
    #     jac = jac.reshape(jac.shape[0], -1)
    #
    #     # prob_over_pbasis_pstate = (
    #     #     prob_array
    #     #     / prob_measurement_basis[None, :, None]
    #     #     / prob_initial_state[:, None, None]
    #     # ).flatten()
    #     fim_element = jax.vmap(lambda x, p: jnp.outer(x, x) / p)(
    #         jac.T, prob_array.flatten()
    #     )
    #     # return fim_element.sum(axis=0)
    #     return jnp.where(~jnp.isinf(fim_element), fim_element, 0).sum(axis=0)

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

    @jax.jit
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

        sum_log_lkl, array_log_lkl_datum = jax.lax.scan(
            f_for_scan, init=jnp.array([0]).astype(jnp.float32), xs=data
        )
        return sum_log_lkl - jnp.max(array_log_lkl_datum)

    def log_lkl_datum_multiple_particles(self, particles, datum: Data):
        loglkl_arr = jax.vmap(self.log_lkl_single_datum, in_axes=(0, None))(
            particles, datum
        )
        return jnp.squeeze(loglkl_arr)

    @jax.jit
    def batch_total_log_lkl(self, data_index, particles, data: Data):
        def f3_for_scan(
            iterstop,
            index,
        ):
            datum = data[index]

            def f1(datum):
                return self.log_lkl_datum_multiple_particles(particles, datum)

            def f2(datum):
                return jnp.zeros(particles.shape[0])

            y = jax.lax.cond(index <= iterstop, f1, f2, datum)
            # y = model.log_lkl_datum_multiple_particles(particles, datum)
            # carry = carry + 1
            return iterstop, y

        return jax.lax.scan(
            f3_for_scan, init=jnp.array(data_index, int), xs=jnp.arange(len(data))
        )[1].sum(0)
