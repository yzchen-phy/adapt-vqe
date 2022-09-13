
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
import operator_pools_pauli
import vqe_methods
from tVQE import *

from openfermion import *


import pyscf_helper

def adapt_vqe_tetris(hamiltonian, pool, reference_ket, file_out, Tetris,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-3,
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{

    ref_energy = reference_ket.T.conj().dot(hamiltonian.dot(reference_ket))[0,0].real
    print(" Reference Energy: %12.8f" %ref_energy)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()
    pool.gradient_print_thresh = theta_thresh

    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []
    curr_state = 1.0*reference_ket
    
    file_out.write("# n_iter -- energy -- index_op \n")
    file_out.flush()

    print(" Now start to grow the ansatz")
    for n_iter in range(0,adapt_maxiter):

        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0

        print(" Check each new operator for coupling")
        next_term = []
        print(" Measure Operator Pool Gradients:")
        sig = hamiltonian.dot(curr_state)
        e_curr = curr_state.T.conj().dot(sig)[0,0]
        var = sig.T.conj().dot(sig)[0,0] - e_curr**2
        uncertainty = np.sqrt(var.real)
        assert(np.isclose(var.imag,0))
        print(" Variance:    %12.8f" %var.real)
        print(" Uncertainty: %12.8f" %uncertainty)

        # list of gradient norms
        grad_list = []

        for oi in range(pool.n_ops):

            gi = pool.compute_gradient_i(oi, curr_state, sig)

            grad_list.append((oi, abs(gi)))

            curr_norm += gi*gi
#            if abs(gi) > abs(next_deriv):
#                next_deriv = gi
#                next_index = oi

        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}

        max_of_gi = next_deriv
        print(" Norm of <[H,A]> = %12.8f" %curr_norm)
        print(" Max  of <[H,A]> = %12.8f" %max_of_gi)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        elif adapt_conver == "var":
            if abs(var) < adapt_thresh:
                #variance
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %12s %18s" %("#","Coeff","Term"))
            for si in range(len(ansatz_ops)):
                print(" %4i %12.8f %s" %(si, parameters[si], ansatz_ops[si]) )
            break

        grad_list.sort(reverse = True, key = lambda x: x[1])
#        for e in grad_list:
#            print(pool.pauli_ops[e[0]], e[1])
        next_index = grad_list[0][0]
        added_op_index = [next_index]

        print(" Add operator %4i" % next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.pauli_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])

        if Tetris == True:
            for ii_op in range(1, pool.n_ops):
                overlap = 0
                for op_ex in added_op_index:
                    if pool.has_overlap(grad_list[ii_op][0], op_ex):
                        overlap += 1
                        break
                if overlap == 0:
                    parameters.insert(0,0)
                    ansatz_ops.insert(0,pool.pauli_ops[grad_list[ii_op][0]])
                    ansatz_mat.insert(0,pool.spmat_ops[grad_list[ii_op][0]])
                    added_op_index.append(grad_list[ii_op][0])
                    print(" Add operator %4i" % grad_list[ii_op][0])

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)


        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %12s %18s" %("#","Coeff","Term"))
        for si in range(len(ansatz_ops)):
            print(" %4i %12.8f %s" %(si, parameters[si], ansatz_ops[si]))
            
        file_out.write("%d \t %12.8f \t %s \n" %(n_iter, trial_model.curr_energy, added_op_index))
#        curr_param_list = "(\t" + join("%12.8f \t" % elem for elem in parameters) + ")"
#        file_out.write("%d \t %12.8f \t %d \t %s \n" %(n_iter, trial_model.curr_energy, next_index, curr_param_list))
        file_out.flush()
        
    return trial_model.curr_energy, curr_state, parameters

# }}}

def adapt_vqe(hamiltonian_op, pool, reference_ket,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-3,
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{

    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    ref_energy = reference_ket.T.conj().dot(hamiltonian.dot(reference_ket))[0,0].real
    print(" Reference Energy: %12.8f" %ref_energy)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()
    pool.gradient_print_thresh = theta_thresh

    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []
    curr_state = 1.0*reference_ket

    print(" Now start to grow the ansatz")
    for n_iter in range(0,adapt_maxiter):

        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0

        print(" Check each new operator for coupling")
        next_term = []
        print(" Measure Operator Pool Gradients:")
        sig = hamiltonian.dot(curr_state)
        e_curr = curr_state.T.conj().dot(sig)[0,0]
        var = sig.T.conj().dot(sig)[0,0] - e_curr**2
        uncertainty = np.sqrt(var.real)
        assert(np.isclose(var.imag,0))
        print(" Variance:    %12.8f" %var.real)
        print(" Uncertainty: %12.8f" %uncertainty)
        for oi in range(pool.n_ops):

            gi = pool.compute_gradient_i(oi, curr_state, sig)

            curr_norm += gi*gi
            if abs(gi) > abs(next_deriv):
                next_deriv = gi
                next_index = oi

        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}

        max_of_gi = next_deriv
        print(" Norm of <[H,A]> = %12.8f" %curr_norm)
        print(" Max  of <[H,A]> = %12.8f" %max_of_gi)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        elif adapt_conver == "var":
            if abs(var) < adapt_thresh:
                #variance
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %12s %18s" %("#","Coeff","Term"))
            for si in range(len(ansatz_ops)):
                opstring = pool.get_string_for_term(ansatz_ops[si])
                print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
            break

        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.fermi_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)


        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %12s %18s" %("#","Coeff","Term"))
        for si in range(len(ansatz_ops)):
            opstring = pool.get_string_for_term(ansatz_ops[si])
            print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
    return trial_model.curr_energy, curr_state, parameters

# }}}

def ucc(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        theta_thresh    = 1e-7,
        pool            = operator_pools.singlet_GSD(),
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule,
                run_scf = 1,
                run_mp2=1,
                run_cisd=0,
                run_ccsd = 0,
                run_fci=1,
                delete_input=1)
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #Thetas
    parameters = [0]*pool.n_ops

    pool.generate_SparseMatrix()

    ucc = UCC(hamiltonian, pool.spmat_ops, reference_ket, parameters)

    opt_result = scipy.optimize.minimize(ucc.energy,
                parameters, options = {'gtol': 1e-6, 'disp':True},
                method = 'BFGS', callback=ucc.callback)
    print(" Finished: %20.12f" % ucc.curr_energy)
    parameters = opt_result['x']
    for p in parameters:
        print(p)

# }}}

def test_random(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-3,
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200,
        pool            = operator_pools.singlet_GSD(),
        psi4_filename   = "psi4_%12.12f"%random.random(),
        seed            = 1
        ):

    # {{{
    random.seed(seed)

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule,
                run_scf = 1,
                run_mp2=1,
                run_cisd=0,
                run_ccsd = 0,
                run_fci=1,
                delete_input=1)
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()

    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []
    curr_state = 1.0*reference_ket

    print(" Now start to grow the ansatz")
    for n_iter in range(0,adapt_maxiter):

        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0

        print(" Check each new operator for coupling")
        next_term = []
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)
        for op_trial in range(pool.n_ops):

            opA = pool.spmat_ops[op_trial]
            com = 2*(curr_state.transpose().conj().dot(opA.dot(sig))).real
            assert(com.shape == (1,1))
            com = com[0,0]
            assert(np.isclose(com.imag,0))
            com = com.real
            opstring = ""
            for t in pool.fermi_ops[op_trial].terms:
                opstring += str(t)
                break

            if abs(com) > adapt_thresh:
                print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com
            if abs(com) > abs(next_deriv):
                next_deriv = com
                next_index = op_trial


        next_index = random.choice(list(range(pool.n_ops)))
        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}

        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" %curr_norm)
        print(" Max  of <[A,H]> = %12.8f" %max_of_com)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %40s %12s" %("#","Term","Coeff"))
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )
            break

        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.fermi_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)


        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(ansatz_ops)):
            s = ansatz_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )

    return
# }}}

def test_lexical(geometry,
        basis           = "sto-3g",
        multiplicity    = 1,
        charge          = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-3,
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200,
        pool            = operator_pools.singlet_GSD(),
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# {{{

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    molecule.filename = psi4_filename
    molecule = openfermionpsi4.run_psi4(molecule,
                run_scf = 1,
                run_mp2=1,
                run_cisd=0,
                run_ccsd = 0,
                run_fci=1,
                delete_input=1)
    pool.init(molecule)
    print(" Basis: ", basis)

    print(' HF energy      %20.16f au' %(molecule.hf_energy))
    print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
    #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
    #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
    print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    #Build p-h reference and map it to JW transform
    reference_ket = scipy.sparse.csc_matrix(
            openfermion.jw_configuration_state(
                list(range(0,molecule.n_electrons)), molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()

    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []
    curr_state = 1.0*reference_ket

    print(" Now start to grow the ansatz")
    for n_iter in range(0,adapt_maxiter):

        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0

        print(" Check each new operator for coupling")
        next_term = []
        print(" Measure commutators:")
        sig = hamiltonian.dot(curr_state)
        for op_trial in range(pool.n_ops):

            opA = pool.spmat_ops[op_trial]
            com = 2*(curr_state.transpose().conj().dot(opA.dot(sig))).real
            assert(com.shape == (1,1))
            com = com[0,0]
            assert(np.isclose(com.imag,0))
            com = com.real
            opstring = ""
            for t in pool.fermi_ops[op_trial].terms:
                opstring += str(t)
                break

            if abs(com) > adapt_thresh:
                print(" %4i %40s %12.8f" %(op_trial, opstring, com) )

            curr_norm += com*com
            if abs(com) > abs(next_deriv):
                next_deriv = com
                next_index = op_trial


        next_index = n_iter % pool.n_ops
        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}

        max_of_com = next_deriv
        print(" Norm of <[A,H]> = %12.8f" %curr_norm)
        print(" Max  of <[A,H]> = %12.8f" %max_of_com)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged:
            print(" Ansatz Growth Converged!")
            print(" Number of operators in ansatz: ", len(ansatz_ops))
            print(" *Finished: %20.12f" % trial_model.curr_energy)
            print(" -----------Final ansatz----------- ")
            print(" %4s %40s %12s" %("#","Term","Coeff"))
            for si in range(len(ansatz_ops)):
                s = ansatz_ops[si]
                opstring = ""
                for t in s.terms:
                    opstring += str(t)
                    break
                print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )
            break

        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.fermi_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)


        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %40s %12s" %("#","Term","Coeff"))
        for si in range(len(ansatz_ops)):
            s = ansatz_ops[si]
            opstring = ""
            for t in s.terms:
                opstring += str(t)
                break
            print(" %4i %40s %12.8f" %(si, opstring, parameters[si]) )

    return
# }}}

def seqGO(hamiltonian_op, pool, reference_ket,
        theta_thresh    = 1e-7,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)
    ref_energy = reference_ket.T.conj().dot(hamiltonian.dot(reference_ket))[0,0].real
    print(" Reference Energy: %12.8f" %ref_energy)

    #Thetas
    parameters = []

    pool.generate_SparseMatrix()
    pool.gradient_print_thresh = theta_thresh

    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz

    print(" Start ADAPT-VQE algorithm")
    op_indices = []
    parameters = []
    curr_state = 1.0*reference_ket

    print(" Now start to grow the ansatz")
    for n_iter in range(0,pool.n_ops):

        print("\n\n\n")
        print(" --------------------------------------------------------------------------")
        print("                         ADAPT-VQE iteration: ", n_iter)
        print(" --------------------------------------------------------------------------")
        next_index = None
        next_deriv = 0
        curr_norm = 0

        print(" Check each new operator for coupling")
        next_term = []
        print(" Measure Operator Pool Gradients:")
        sig = hamiltonian.dot(curr_state)
        e_curr = curr_state.T.conj().dot(sig)[0,0]
        var = sig.T.conj().dot(sig)[0,0] - e_curr**2
        uncertainty = np.sqrt(var.real)
        assert(np.isclose(var.imag,0))
        print(" Variance:    %12.8f" %var.real)
        print(" Uncertainty: %12.8f" %uncertainty)
        for oi in range(pool.n_ops):

            gi = pool.compute_gradient_i(oi, curr_state, sig)

            curr_norm += gi*gi
            if abs(gi) > abs(next_deriv):
                next_deriv = gi
                next_index = oi

        curr_norm = np.sqrt(curr_norm)

        min_options = {'gtol': theta_thresh, 'disp':False}

        max_of_gi = next_deriv
        print(" Norm of <[H,A]> = %12.8f" %curr_norm)
        print(" Max  of <[H,A]> = %12.8f" %max_of_gi)

        # converged = False
        # if adapt_conver == "norm":
        #     if curr_norm < adapt_thresh:
        #         converged = True
        # elif adapt_conver == "var":
        #     if abs(var) < adapt_thresh:
        #         #variance
        #         converged = True
        # else:
        #     print(" FAIL: Convergence criterion not defined")
        #     exit()
        #
        # if converged:
        #     print(" Ansatz Growth Converged!")
        #     print(" Number of operators in ansatz: ", len(ansatz_ops))
        #     print(" *Finished: %20.12f" % trial_model.curr_energy)
        #     print(" -----------Final ansatz----------- ")
        #     print(" %4s %12s %18s" %("#","Coeff","Term"))
        #     for si in range(len(ansatz_ops)):
        #         opstring = pool.get_string_for_term(ansatz_ops[si])
        #         print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
        #     break

        print(" Add operator %4i" %next_index)
        parameters.insert(0,0)
        ansatz_ops.insert(0,pool.fermi_ops[next_index])
        ansatz_mat.insert(0,pool.spmat_ops[next_index])

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)


        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                options = min_options, method = 'BFGS', callback=trial_model.callback)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(" -----------New ansatz----------- ")
        print(" %4s %12s %18s" %("#","Coeff","Term"))
        for si in range(len(ansatz_ops)):
            opstring = pool.get_string_for_term(ansatz_ops[si])
            print(" %4i %12.8f %s" %(si, parameters[si], opstring) )
    return trial_model.curr_energy, curr_state, parameters



def Make_S2(n_orb):
# {{{
    ap =scipy.sparse.csc_matrix(np.array([[0, 0], [1, 0]]))    #creation operator
    am =scipy.sparse.csc_matrix( np.array([[0, 1], [0, 0]])) #annihilation operator
    no =scipy.sparse.csc_matrix( np.array([[0, 0], [0, 1]]))     #number operator
    ho =scipy.sparse.csc_matrix( np.array([[1, 0], [0, 0]]))     #hole operator
    I2 =scipy.sparse.csc_matrix( np.array([[1, 0], [0, 1]]))     #identity operator
    Iz =scipy.sparse.csc_matrix( np.array([[1, 0], [0, -1]]))    #pauli z operat
    S2 =scipy.sparse.csc_matrix( np.zeros((4**n_orb,4**n_orb)))
    s2 =scipy.sparse.csc_matrix( np.array([[0,0],[0,0.75]]))

    for i in range(0,n_orb):
        bfor  = 2*i
        aftr  = 2*n_orb-2*i-2

        Ia = np.eye(np.power(2,bfor))
        Ib = np.eye(np.power(2,aftr))
        a_temp = scipy.sparse.kron(s2,I2)
        b_temp = scipy.sparse.kron(I2,s2)
        S2a = scipy.sparse.kron(Ia,scipy.sparse.kron(a_temp,Ib))
        S2b = scipy.sparse.kron(Ia,scipy.sparse.kron(b_temp,Ib))

        S2 += abs(S2a -S2b)


        for j in range(i+1,n_orb):

            intr = 2*j-2*i-2
            aftr = 2*n_orb-2*j-2

            Ib = np.eye(np.power(2,intr))
            Zb = np.eye(1)
            for k in range(2*i+2,2*j):
                Zb = scipy.sparse.kron(Zb,Iz)

            Ic = np.eye(np.power(2,aftr))
            Zc = np.eye(1)
            for k in range(2*j,2*n_orb-2):
                Zc = scipy.sparse.kron(Zc,Iz)

            assert(Zc.shape == Ic.shape)
            assert(Zb.shape == Ib.shape)


            Sptemp = scipy.sparse.kron(ap,am)
            Smtemp = scipy.sparse.kron(am,ap)
            ANtemp = scipy.sparse.kron(no,I2)
            BNtemp = scipy.sparse.kron(I2,no)

            ##CASE A
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(Sptemp,scipy.sparse.kron(Ib,scipy.sparse.kron(Smtemp,Ic))))
            S2  +=  (aiaj)

            ##CASE B
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(Smtemp,scipy.sparse.kron(Ib,scipy.sparse.kron(Sptemp,Ic))))
            S2  +=  (aiaj)

            ##CASE C
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(ANtemp,scipy.sparse.kron(Ib,scipy.sparse.kron(BNtemp,Ic))))
            S2  -= 0.5 * (aiaj)

            ##CASE D
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(BNtemp,scipy.sparse.kron(Ib,scipy.sparse.kron(ANtemp,Ic))))
            S2  -= 0.5 * (aiaj)

            ##CASE E
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(ANtemp,scipy.sparse.kron(Ib,scipy.sparse.kron(ANtemp,Ic))))
            S2  += 0.5 * (aiaj)

            ##CASE F
            aiaj = scipy.sparse.kron(Ia,scipy.sparse.kron(BNtemp,scipy.sparse.kron(Ib,scipy.sparse.kron(BNtemp,Ic))))
            S2  += 0.5 * (aiaj)

    return scipy.sparse.csc_matrix(S2)
    # }}}





if __name__== "__main__":
    r = 1.5
    #geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]
    geometry = [('H',  (0, 0, 0)),
                ('Li', (0, 0, r*2.39))]
    #geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r)), ('H', (0,0,5*r)), ('H', (0,0,6*r))]


    charge = 0
    spin = 0
    basis = 'sto-3g'

#    geometry = [('Sc', (0,0,0))]
#    charge = 1
#    spin = 0
#    #[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis)
#    mo_order = []
#    mo_order.extend(range(0,9))
#    mo_order.extend(range(9,10))
#    mo_order.extend(range(13,18))
#    mo_order.extend(range(10,13))
#    print(" mo_order: ", mo_order)
#    [n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis,n_frzn_occ=9,
#            n_act=6, mo_order=mo_order)
    [n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis)

    print(" n_orb: %4i" %n_orb)
    print(" n_a  : %4i" %n_a)
    print(" n_b  : %4i" %n_b)

    sq_ham = pyscf_helper.SQ_Hamiltonian()
    sq_ham.init(h, g, C, S)
    print(" HF Energy: %12.8f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

    fermi_ham  = sq_ham.export_FermionOperator()

    hamiltonian = openfermion.transforms.get_sparse_operator(fermi_ham)

    s2 = Make_S2(n_orb)

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

    pool = operator_pools.singlet_SD()
    pool.init(n_orb, n_occ_a=n_a, n_occ_b=n_b, n_vir_a=n_orb-n_a, n_vir_b=n_orb-n_b)

    [e,v,params] = vqe_methods.adapt_vqe(fermi_ham, pool, reference_ket, theta_thresh=1e-9)
    # [e,v,params] = vqe_methods.seqGO(fermi_ham, pool, reference_ket, theta_thresh=1e-9)

    print(" Final ADAPT-VQE energy: %12.8f" %e)
    print(" <S^2> of final state  : %12.8f" %(v.conj().T.dot(s2.dot(v))[0,0].real))
