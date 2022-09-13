# Usage: trial_tetris.py <multiples of bond r>

import sys
sys.path.insert(0, '../src')
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import scipy
import vqe_methods 
import operator_pools_pauli
import pyscf_helper 

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, molden, cc
from pyscf.cc import ccsd

import openfermion 
from openfermion import *
from tVQE import *
    

def main():
    r = float(sys.argv[1])
    geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r)), ('H', (0,0,5*r)), ('H', (0,0,6*r))]
    Tetris = False

    #filename = "data_qubit/H6_%1.1f_reg.txt" % (r)
    filename = "data_QEB/H6_%1.1f_reg.txt" % (r)

    file_out = open(filename, "w")

    charge = 0
    spin = 0
    basis = 'sto-3g'

    [n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis)

    print(" n_orb: %4i" %n_orb)
    print(" n_a  : %4i" %n_a)
    print(" n_b  : %4i" %n_b)

    sq_ham = pyscf_helper.SQ_Hamiltonian()
    sq_ham.init(h, g, C, S)
    print(" HF Energy: %12.8f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

    fermi_ham  = sq_ham.export_FermionOperator()

    hamiltonian = openfermion.transforms.get_sparse_operator(fermi_ham)

    s2 = vqe_methods.Make_S2(n_orb)

    #build reference configuration
    occupied_list = []
    for i in range(n_a):
        occupied_list.append(i*2)
    for i in range(n_b):
        occupied_list.append(i*2+1)

    print(" Build reference state with %4i alpha and %4i beta electrons" %(n_a,n_b), occupied_list)
    reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(occupied_list, 2*n_orb)).transpose()

    [e,v] = scipy.sparse.linalg.eigsh(hamiltonian.real,1,which='SA',v0=reference_ket.todense())
    for ei in range(len(e)):
        S2 = v[:,ei].conj().T.dot(s2.dot(v[:,ei]))
        print(" State %4i: %12.8f au  <S2>: %12.8f" %(ei,e[ei]+E_nuc,S2))
    
    fermi_ham += FermionOperator((),E_nuc)
    pyscf.molden.from_mo(mol, "full.molden", sq_ham.C)

    file_out.write("#E_gs: \t %12.8f \n" % (e[0]+E_nuc))

    hamiltonian = openfermion.transforms.get_sparse_operator(fermi_ham)

    #pool = operator_pools_pauli.Pauli_qubit()
    pool = operator_pools_pauli.Pauli_QEB()
    pool.init(n_orb)

    [e,v,params] = vqe_methods.adapt_vqe_tetris(hamiltonian, pool, reference_ket, file_out, Tetris, adapt_maxiter=200, adapt_thresh=1e-7, theta_thresh=1e-10)

    print(" Final ADAPT-VQE energy: %12.8f" %e)
    print(" <S^2> of final state  : %12.8f" %(v.conj().T.dot(s2.dot(v))[0,0].real))

    file_out.close()

if __name__== "__main__":
    main()
