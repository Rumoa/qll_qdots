import numpy as np
import joblib
import qutip as qu
from process_representations import *

l_true = joblib.load("L_num.job")

delta = 0.12739334807998307
Omega = 0.5

A = np.array([[1, 0], [0, 0]])
sigmap = np.array([[0, 1], [0, 0]])
sigman = np.array([[0, 0], [1, 0]])


H0 = delta * A + Omega / 2 * (sigmap + sigman)

# print(l_true)

# print(compare_ab(qu.to_chi(l_true), to_chi_from_super(l_true.full())))

print(qu.to_chi(l_true).full().round(3))
print(to_chi_from_super(l_true.full()).round(3))
