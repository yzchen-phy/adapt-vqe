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

# for p in range(h1.shape[0]-1):
#     H += t*FermionOperator(((p,1),(p+1,0)))
#     H += U*FermionOperator(((p,1),(p,0),(p+1,1),(p+1,0)))
H = openfermion.FermionOperator((),0)
H += FermionOperator(((0,1),(1,0)),t)
H += FermionOperator(((1,1),(2,0)),t)
# H += FermionOperator(((2,1),(3,0)),t)
N = openfermion.FermionOperator()
for i in range(3):
    N += FermionOperator(((i,1),(i,0)))
H += hermitian_conjugated(H)
H = normal_ordered(H)
print(H)

hamiltonian = openfermion.transforms.get_sparse_operator(H)
N = openfermion.transforms.get_sparse_operator(N)
[e,v] = np.linalg.eigh(hamiltonian.todense())
for ei in range(len(e)):
    # n = np.linalg.multi_dot([v[:,ei].conj().T,N,v[:,ei]])
    vi = v[:,ei]
    n = vi.conj().T.dot(N.dot(vi))[0,0]
    print(" State %4i: %12.8f au: Number %12.8f" %(ei,e[ei],n))

for vi in range(len(v[:,1])):
    print(" %4i %12.8f i%12.8f" %(vi+1, v[vi,0].real, v[vi,0].imag))

mu = 1e4
print("Now do 4 site")
# 0 -> 0: 1->1,2: 2->3
H = openfermion.FermionOperator((),0)
H += FermionOperator(((0,1),(1,0),(2,0)),t)
H += FermionOperator(((1,1),(2,1),(3,0)),t)
# H += FermionOperator(((4,1),(5,0)),t)
H += FermionOperator(((1,0),(1,1),(2,1),(2,0)),mu)
H += FermionOperator(((1,1),(1,0),(2,0),(2,1)),mu)
# H += FermionOperator(((3,0),(3,1),(4,1),(4,0)),mu)
H += hermitian_conjugated(H)
H = normal_ordered(H)

N = openfermion.FermionOperator()
for i in range(4):
    N += FermionOperator(((i,1),(i,0)))

print(H)
hamiltonian = openfermion.transforms.get_sparse_operator(H)
N = openfermion.transforms.get_sparse_operator(N)
[e,v] = np.linalg.eigh(hamiltonian.todense())
for ei in range(len(e)):
    vi = v[:,ei]
    n = vi.conj().T.dot(N.dot(vi))[0,0]
    print(" State %4i: %12.8f au: Number %12.8f" %(ei,e[ei],n))
for vi in range(len(v[:,0])):
    print(" %4i %12.8f i%12.8f" %(vi+1, v[vi,0].real, v[vi,0].imag))
