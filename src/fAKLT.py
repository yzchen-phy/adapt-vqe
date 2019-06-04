import scipy
import openfermion
from openfermion import *
import numpy as np
import copy
import random
import sys

"""
o--o--o--o  --> (o--o--o-)(-o--o--o)
"""
h1 = np.zeros([4,4])
h2 = np.zeros([4,4,4,4])
H = openfermion.FermionOperator()

t=-1
U=1

for p in range(h1.shape[0]-1):
    H += t*FermionOperator(((p,1),(p+1,0)))
    H += U*FermionOperator(((p,1),(p,0),(p+1,1),(p+1,0)))
H += hermitian_conjugated(H)
H = normal_ordered(H)

hamiltonian = openfermion.transforms.get_sparse_operator(H)
[e,v] = np.linalg.eigh(hamiltonian.todense())
for ei in range(len(e)):
    print(" State %4i: %12.8f au" %(ei,e[ei]))


print(H)
