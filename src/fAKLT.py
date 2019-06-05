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
H += FermionOperator(((1,1),(2,0)),t)*.8
H += FermionOperator(((2,1),(0,0)),t)*.9
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

mu = 1e5
print("Now do 6 site")
n = []
m = []
for ni in range(6):
    n.append(FermionOperator(((ni,1),(ni,0))))
    m.append(FermionOperator(((ni,0),(ni,1))))

c0 = FermionOperator(((0,1)))
c1 = FermionOperator(((1,1)))
c2 = FermionOperator(((2,1)))
c3 = FermionOperator(((3,1)))
c4 = FermionOperator(((4,1)))
c5 = FermionOperator(((5,1)))
a0 = hermitian_conjugated(c0)
a1 = hermitian_conjugated(c1)
a2 = hermitian_conjugated(c2)
a3 = hermitian_conjugated(c3)
a4 = hermitian_conjugated(c4)
a5 = hermitian_conjugated(c5)

cc0 = c0*c5
cc1 = c1*c2
cc2 = c3*c4
cc0 += mu*(c0*a5+a0*c5)
cc1 += mu*(c1*a2+a1*c2)
cc2 += mu*(c3*a4+a3*c4)
# cc0 -= hermitian_conjugated(cc0)
# cc1 -= hermitian_conjugated(cc1)
# cc2 -= hermitian_conjugated(cc2)
aa0 = hermitian_conjugated(cc0)
aa1 = hermitian_conjugated(cc1)
aa2 = hermitian_conjugated(cc2)

# print("commutator")
# print(normal_ordered(cc1*aa0+aa0*cc1))
# exit()

# 0 -> 0: 1->1,3: 2->3
H = openfermion.FermionOperator((),0)
H += t*(cc0*aa1 + .8*cc1*aa2 + .9*cc2*aa0)
# H += mu*(n[0]*m[5] + m[0]*n[5] + n[1]*m[2] + m[1]*n[2] + n[3]*m[4] + m[3]*n[4])
# H += t*(c0*a1 + c1*a2 + c2*a0)
H += hermitian_conjugated(H)
H = normal_ordered(H)

N = openfermion.FermionOperator()
for i in range(6):
    N += FermionOperator(((i,1),(i,0)))

print(H)
hamiltonian = openfermion.transforms.get_sparse_operator(H)
N = openfermion.transforms.get_sparse_operator(N)
[e,v] = np.linalg.eigh(hamiltonian.todense())
for ei in range(len(e)):
    vi = v[:,ei]
    n = vi.conj().T.dot(N.dot(vi))[0,0]
    if abs(e[ei]) > 1000:
        continue
    print(" State %4i: %12.8f au: Number %12.8f" %(ei,e[ei],n))
# for vi in range(len(v[:,0])):
#     print(" %4i %12.8f i%12.8f" %(vi+1, v[vi,0].real, v[vi,0].imag))
