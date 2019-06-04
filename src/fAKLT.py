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
H3 = openfermion.FermionOperator()

t=-1
U=1

for p in range(h1.shape[0]-1):
    H3 += t*FermionOperator(((2*p,1),(2*p+2,0)))
    H3 += t*FermionOperator(((2*p+1,1),(2*p+3,0)))
    H3 += U*FermionOperator(((2*p,1),(2*p,0),(2*p+1,1),(2*p+1,0)))
H3 += hermitian_conjugated(H3)
H3 = normal_ordered(H3)
hamiltonian = openfermion.transforms.get_sparse_operator(H3)
[e,v] = np.linalg.eigh(hamiltonian.todense())
for ei in range(len(e)):
    print(" State %4i: %12.8f au" %(ei,e[ei]))


print(H3)
