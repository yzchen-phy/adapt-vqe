import scipy
import openfermion
import openfermionpsi4
import os
import numpy as np
import copy
import random 
import sys

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, molden, cc
from pyscf.cc import ccsd

import operator_pools
import vqe_methods
from tVQE import *

from openfermion import *


import pyscf_helper



charge = 1
spin = 0
basis = 'cc-pvdz'

geometry = [('Sc', (0,0,0))]

mo_order = []
mo_order.extend(range(0,11))
mo_order.extend(range(14,18))
mo_order.extend(range(11,14))
mo_order.extend(range(18,43))

print(" mo_order: ", mo_order)
[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis,n_frzn_occ=9,
        n_act=6, mo_order=mo_order)

print(" n_orb: %4i" %n_orb)
print(" n_a  : %4i" %n_a)
print(" n_b  : %4i" %n_b)

sq_ham = pyscf_helper.SQ_Hamiltonian()
sq_ham.init(h, g, C, S)
print(" HF Energy: %12.8f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

fermi_ham  = sq_ham.export_FermionOperator()

hamiltonian = openfermion.transforms.get_sparse_operator(fermi_ham)

s2 = vqe_methods.Make_S2(n_orb)

#n_a += 1

#build reference configuration
occupied_list = []
for i in range(n_a):
    occupied_list.append(i*2)
for i in range(n_b):
    occupied_list.append(i*2+1)

print(" Build reference state with %4i alpha and %4i beta electrons" %(n_a,n_b), occupied_list)
reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(occupied_list, 2*n_orb)).transpose()

print(" Do exact diagonalization of full Hilbert space Hamiltonian")
[e,v] = scipy.sparse.linalg.eigsh(hamiltonian.real,180,which='SA',v0=reference_ket.todense())
for ei in range(len(e)):
    S2 = v[:,ei].conj().T.dot(s2.dot(v[:,ei]))
    print(" FCI State %4i: %12.8f au  <S2>: %12.8f" %(ei,e[ei]+E_nuc,S2))
fermi_ham += FermionOperator((),E_nuc)
pyscf.molden.from_mo(mol, "full.molden", sq_ham.C)

pool = operator_pools.singlet_GSD()
pool.init(n_orb)

[e,v,params] = vqe_methods.adapt_vqe(fermi_ham, pool, reference_ket, adapt_thresh=1e-6, theta_thresh=1e-9)

print(" Final ADAPT-VQE energy: %12.8f" %e)
print(" <S^2> of final state  : %12.8f" %(v.conj().T.dot(s2.dot(v))[0,0].real))
