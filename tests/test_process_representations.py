import logging
from qdots_qll.process_representations import (
    compare_ab,
    to_chi_from_choi,
    to_choi_from_super,
    to_super_from_chi,
    to_chi_from_super,
    to_super_from_choi,
)
import qutip as qu

logger = logging.getLogger("__process_representations__")
logger.setLevel(logging.INFO)


seed = 1
example_process = qu.rand_super(2, seed=seed)

example_xij = qu.rand_herm(N=4, seed=seed + 1)


def test_to_choi_from_super():
    assert compare_ab(
        to_choi_from_super(example_process.full()), qu.to_choi(example_process)
    ).all()


def test_to_super_from_choi():
    assert compare_ab(
        example_process.full(),
        to_super_from_choi(qu.to_choi(example_process).full()),
    ).all()


def test_to_chi_from_super():
    assert compare_ab(
        to_chi_from_super(to_super_from_chi(example_xij.full())),
        example_xij.full(),
    ).all()


def test_to_chi_from_choi():
    assert compare_ab(
        to_chi_from_choi(
            to_choi_from_super(to_super_from_chi(example_xij.full()))
        ),
        example_xij.full(),
    ).all()
