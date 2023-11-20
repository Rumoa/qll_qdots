from qdots_qll.all_funcs import vec, dag, v_trace, v_evolve, compute_p
import qutip as qu
import numpy as np

seed = 1
d = 2
example_rho = qu.rand_dm_ginibre(N=d, seed=seed)


def test_vec():
    assert np.isclose(
        qu.operator_to_vector(example_rho).full(), vec(example_rho.full())
    ).all()


def test_dag():
    assert np.isclose(example_rho.dag(), dag(example_rho.full())).all()


def test_v_trace():
    assert np.isclose(
        np.trace(example_rho.full()), v_trace(vec(example_rho.full()))
    )


def test_v_evolve():
    example_L = qu.rand_super(N=d, seed=seed)
    t = 0.4582

    A = ((t * example_L).expm() * qu.operator_to_vector(example_rho)).full()
    B = v_evolve(t, example_L.full(), vec(example_rho.full()))

    assert np.isclose(A / v_trace(A), B).all()


def test_compute_p():
    kr = np.array(qu.to_kraus(qu.rand_super_bcsz(N=2, seed=seed)))
    povm = np.array(
        [np.einsum("jk, kl", np.conjugate(i.T), i) for i in list(kr)]
    )
    p1 = np.trace(povm[1] @ example_rho.full())
    p2 = compute_p(vec(example_rho.full()), vec(povm[1]))
    assert np.isclose(p1, p2)
