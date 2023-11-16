import jax
import numpy as np
import jax.numpy as jnp
import qutip as qu

d = 2  # For now, we will stick to 1 qubit


N = d  # dimension of Hilbert Space
G = np.array([qu.identity(N), qu.sigmax(), qu.sigmay(), qu.sigmaz()]) / np.sqrt(N)


def compare_ab(A, B):
    return np.isclose(A, B)


def to_choi_from_super(superop):
    basis = np.expand_dims(np.identity(2), axis=2)
    # column is basis[i]
    suma = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for L in range(2):
                    suma = suma + np.kron(basis[L].T, basis[j].T) @ superop @ np.kron(
                        basis[k], basis[i]
                    ) * np.kron(basis[i], basis[j]) @ np.kron(basis[k].T, basis[L].T)

    return suma


def to_super_from_choi(choi):
    return to_choi_from_super(choi)


def to_chi_from_choi(choi):  # I think this is wrong. we need to correct it
    omega_ket = np.array(
        [np.kron(qu.basis(2, i), qu.basis(2, i)) for i in range(2)]
    ) / np.sqrt(2)

    chimn = np.zeros([4, 4], dtype=np.complex64)

    for i in range(4):
        for j in range(4):
            suma = 0
            for k in range(2):
                for L in range(2):
                    suma = suma + (
                        np.conjugate(omega_ket[L].T)
                        @ np.conjugate(
                            np.kron(
                                np.identity(2),
                                G[i],
                            ).T
                        )
                        @ choi
                        @ np.kron(np.identity(2), G[j])
                        @ omega_ket[k]
                    )
            chimn[i, j] = suma

    return 2 * chimn


def to_super_from_chi(xi):
    # return np.einsum("ij,ijkl -> kl ",  xi, G_ijkl)
    a = np.zeros([4, 4], dtype=np.complex64)
    for i in range(4):
        for j in range(4):
            a = a + xi[i][j] * np.kron(np.transpose(G[j]), G[i])
    return a


def to_chi_from_super(superopmat):  # this is correct.
    aux = np.zeros([4, 4], dtype=np.complex64)

    for alpha in range(4):
        for beta in range(4):
            suma = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for L in range(2):
                            value = (
                                np.trace(
                                    G[beta].T
                                    @ (qu.basis(2, i) * qu.basis(2, k).dag()).full()
                                )
                                * np.trace(
                                    G[alpha]
                                    @ (qu.basis(2, j) * qu.basis(2, L).dag()).full()
                                )
                                * np.kron(
                                    qu.basis(2, i).dag().full(),
                                    qu.basis(2, j).dag().full(),
                                )
                                @ superopmat
                                @ np.kron(qu.basis(2, k).full(), qu.basis(2, L).full())
                            )
                            suma = suma + value
            aux[alpha, beta] = suma
    return aux
