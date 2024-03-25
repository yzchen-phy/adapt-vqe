import scipy
import openfermion
import networkx as nx
import os
import numpy as np
import copy
import random
import sys
import pickle
import math
from scipy.linalg import norm

import operator_pools_qaoa as operator_pools
from tVQE import *

from openfermion import *

###########################################################################
# Original
###########################################################################

def run(n,
         f,
         f_mix,
         f_ent,
         f_ent_m,
         f_Fid,
         f_fin,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
         layer = 1,
         pool=operator_pools.qaoa(),
         psi_ref = '+',         
         init_para = 0.01,
         structure = 'qaoa',
         selection = 'NA',
         rand_ham = 'False',
         opt_method = 'NM',
         landscape = False,
         landscape_after = False,
         resolution = 100
         ):
    # {{{

    hamiltonian = pool.cost_mat[0] * 1j
    # print('hamiltonian:',hamiltonian)

    ## Check about the degeneracy
    tolerance = 1e-12	

    H = np.zeros((2**n, 2**n))
    H = hamiltonian.real
    h = H.diagonal()
    print ('spectrum:', h)
    
    hard_min = np.min(h)
    degenerate_indices = np.argwhere(h < hard_min + tolerance).flatten()
    GS_energy = hard_min

    deg_manifold = []
    for ind in degenerate_indices:
    	deg_state = np.zeros(h.shape)
    	deg_state[ind] = 1.0
    	deg_manifold.append(deg_state) 

    print('deg_manifold_length:', len(deg_manifold))

    ## Calculate the ground state energy
    #w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    #GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    #GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )
       
    # Start from |+> states
    if psi_ref == '+':
        reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )
    
    # Start from symmetry breaking state:
    state_ref = np.full((2**n, 1), 0.0+0.0j)
    # State |0++...>
    if psi_ref == '0':
        for ii in range(2**(n-1)):
            state_ref[ii] = 1/np.sqrt(2**(n-1))
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
        
    # Or state |1++...>
    if psi_ref == '1':
        for ii in range(2**(n-1)):
            state_ref[ii+2**(n-1)] = 1/np.sqrt(2**(n-1))
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
        
    # Or state |(+i)++...>
    if psi_ref == 'i':
        for ii in range(2**(n-1)):
            state_ref[ii] = 1/np.sqrt(2**n)
        for ii in range(2**(n-1)):
            state_ref[ii+2**(n-1)] = 1j/np.sqrt(2**n)
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
    
    reference_bra = reference_ket.transpose().conj()
    E = reference_bra.dot(hamiltonian.dot(reference_ket))[0,0].real

    # Thetas 
    parameters = []
    
    print(" structure :", structure)
    print(" selection :", selection)
    print(" initial parameter:", init_para)
    print(" optimizer:", opt_method)
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []

    min_options = {'gtol': theta_thresh, 'disp':False}

    for p in range(0, layer):
        print(" --------------------------------------------------------------------------")
        print("                                  layer: ", p+1)
        print(" --------------------------------------------------------------------------")
        
        if structure == 'qaoa':
            ansatz_ops.insert(0, pool.cost_ops[0])
            ansatz_mat.insert(0, pool.cost_mat[0])
            parameters.insert(0, init_para)

        if selection == 'NA':
            ansatz_ops.insert(0, pool.mixer_ops[0])
            ansatz_mat.insert(0, pool.mixer_mat[0])
            parameters.insert(0, init_para)
            
            f_mix.write("%d \t %d \n" % (p, 0))
            f_mix.flush()

        if selection == 'grad':
            trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

            curr_state = trial_model.prepare_state(parameters)
            #print("current state: ", curr_state)

            sig = hamiltonian.dot(curr_state)

            # Find the operators in the pool causing the largest descent
            next_index = 0
            next_deriv = 0
            for op_trial in range(pool.n_ops):
                opA = pool.spmat_ops[op_trial]
                com = 2 * (curr_state.transpose().conj().dot(opA.dot(sig))).real
                assert (com.shape == (1, 1))
                com = com[0, 0]
                assert (np.isclose(com.imag, 0))
                com = com.real
                if abs(com) > abs(next_deriv) + adapt_thresh:
                    next_deriv = abs(com)
                    next_index = op_trial
                elif abs(com) > abs(next_deriv) - adapt_thresh:
                    next_index = random.choice([next_index, op_trial])
            
            if next_deriv < adapt_thresh:
                f_mix.write("#Ansatz Growth Converged!")
                f_mix.flush()
                break
            new_op = pool.pool_ops[next_index]
            new_mat = pool.spmat_ops[next_index]
    
            print(" Add operator %4i" % next_index)
            f_mix.write("%d \t %d \n" % (p, next_index))
            f_mix.flush()

            parameters.insert(0, 0)
            #parameters.insert(0, init_para)
            ansatz_ops.insert(0, new_op)
            ansatz_mat.insert(0, new_mat)

            if landscape == True:
                lattice = np.arange(-math.pi, math.pi, 2*math.pi/resolution)
                land = np.zeros(shape=(len(lattice), len(lattice)))
                for i in range(len(lattice)):
                    for j in range(len(lattice)):
                        para_land = parameters.copy()
                        para_land[0] = lattice[i] # gamma, cost parameter
                        # print(para_land)
                        para_land.insert(0, lattice[j])
                        # print(para_land)
                        # print("length of parameters", len(parameters)) 
                        # print("length of para_land", len(para_land))
                        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, para_land)
                        land_state = trial_model.prepare_state(para_land)
                        land[i, j] = trial_model.energy(para_land)
                pickle.dump(land, open('./landscape_%s_%s_%s_%d.p' %(selection, init_para, opt_method, p),'wb'))


        # With the operator for this layer chosen, optimize the parameters
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        if opt_method == 'NM':    
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
                                                         method='Nelder-Mead', callback=None)
#            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
#                                                         method='Nelder-Mead', callback=trial_model.callback)
#            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
#                                                         method='Nelder-Mead', options={'maxiter':1000}, callback=None)
#            minimizer_kwargs = {"method": "Nelder-Mead"}
#            opt_result = scipy.optimize.basinhopping(trial_model.energy, parameters,
#                                                        minimizer_kwargs=minimizer_kwargs, niter=50)

        print(opt_result['message'])
        
        if opt_method == 'BFGS':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                                                            options = min_options, method = 'BFGS', callback=trial_model.callback)
        if opt_method == 'Cobyla':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, 
                                                  method = 'Cobyla')
        
        parameters = list(opt_result['x'])
        fin_energy = trial_model.curr_energy.copy()

        # Calculate the entanglement spectrum and entropy
        spec = trial_model.entanglement_spectrum(parameters)
        # Equal bipartition
        entropy = trial_model.entanglement_entropy(parameters)
        # Bipartition 1-(n-1)
        entropy1 = trial_model.entanglement_entropy1(parameters)
        f_ent.write("%d" % (p))
        for e in spec:
            f_ent.write("\t %20.12f" % (e))
        f_ent.write("\t %20.12f" % (entropy))
        f_ent.write("\t %20.12f \n" % (entropy1))
        f_ent.flush()
        
        # Calculate the entanglement spectrum and entropy after projective measurement on qubit 0
        spec = trial_model.entanglement_spectrum_m(parameters)
        # Equal bipartition
        entropy = trial_model.entanglement_entropy_m(parameters)
        # Bipartition 1-(n-1)
        entropy1 = trial_model.entanglement_entropy1_m(parameters)
        f_ent_m.write("%d" % (p))
        for e in spec:
            f_ent_m.write("\t %20.12f" % (e))
        f_ent_m.write("\t %20.12f" % (entropy))
        f_ent_m.write("\t %20.12f \n" % (entropy1))
        f_ent_m.flush()

        print(" Finished: %20.12f" % fin_energy)
        #print(' maxcut objective:', trial_model.curr_energy + pool.shift)
        print(' Error:', trial_model.curr_energy - GS_energy.real)
        print(' Params:', parameters)
        #print('Entanglement_entropy:', entropy)
        sys.stdout.flush()
        
        f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real + pool.shift.real)))
        #f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))
        #f_overlap.write("%d    %20.15f \n" % (p, overlap))
        f.flush()
        
        state_fin = trial_model.prepare_state(parameters)
        overlap = 0.0
        for ii in degenerate_indices:
            overlap += abs(state_fin[ii, 0])**2
        f_Fid.write("%d    %20.15f \n" % (p, overlap))
        f_Fid.flush() 

        #state_fin = trial_model.prepare_state(parameters)
        #(row, col, val) = scipy.sparse.find(state_fin)
        #print('optimal state at layer %d' % (p))
        #for ii in range(len(row)):
        #    print ('%s\t%s' % (row[ii], val[ii]))

    #print("final state: ", trial_model.prepare_state(parameters))
    #state_fin = trial_model.prepare_state(parameters)
    #(row, col, val) = scipy.sparse.find(state_fin)
    #for ii in range(len(row)):
    #    print ('%s\t%s' % (row[ii], val[ii]))
    
    #f_mix.write("# Final ansatz: ")
    #for ind in range(len(ansatz_ops)):
    #    print(" %4s %20f %10s" % (ansatz_ops[ind], parameters[ind], ind))
    #    print("")
    #    if (ind % 2) == 0:
    #        f_mix.write("%s %20f %s & \t" % (ansatz_ops[ind], parameters[ind], ind))
    #    if (ind % 2) == 1:
    #        f_mix.write("%s %20f %s & \t" % ('1j H_C', parameters[ind], ind))
    #f_mix.flush()
    
    for ind in range(layer):
        ansatz_mat_temp = ansatz_mat[2*(layer-ind-1):2*layer]
        param_temp = parameters[2*(layer-ind-1):2*layer]
        model_temp = tUCCSD(hamiltonian, ansatz_mat_temp, reference_ket, param_temp)
        
        # Calculate the entanglement spectrum and entropy
        spec = model_temp.entanglement_spectrum(param_temp)
        # Equal bipartition
        entropy = model_temp.entanglement_entropy(param_temp)
        # Bipartition 1-(n-1)
        entropy1 = model_temp.entanglement_entropy1(param_temp)
        error = (GS_energy.real - model_temp.energy(param_temp)) / GS_energy.real 
        
        f_fin.write("%d" % (ind))
        for e in spec:
            f_fin.write("\t %20.12f" % (e))
        f_fin.write("\t %20.12f" % (entropy))
        f_fin.write("\t %20.12f" % (entropy1))
        f_fin.write("\t %20.12f \n" % (error))
        f_fin.flush()

###########################################################################
# Tetris with parameter 0 individual / 1 simultaneous rotations
###########################################################################

def run_tetris(n,
         f,
         f_mix,
         f_ent,
         f_ent_m,
         f_Fid,
         f_fin,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
         layer = 1,
         pool=operator_pools.qaoa(),
         psi_ref = '+',
         init_para = 0.01,
         structure = 'qaoa',
         selection = 'NA',
         Tetris = 0,
         rand_ham = 'False',
         opt_method = 'NM',
         landscape = False,
         landscape_after = False,
         resolution = 100
         ):
    # {{{

    hamiltonian = pool.cost_mat[0] * 1j
    # print('hamiltonian:',hamiltonian)

    ## Check about the degeneracy
    tolerance = 1e-12	

    H = np.zeros((2**n, 2**n))
    H = hamiltonian.real
    h = H.diagonal()
    print ('spectrum:', h)
    
    hard_min = np.min(h)
    degenerate_indices = np.argwhere(h < hard_min + tolerance).flatten()
    GS_energy = hard_min

    deg_manifold = []
    for ind in degenerate_indices:
    	deg_state = np.zeros(h.shape)
    	deg_state[ind] = 1.0
    	deg_manifold.append(deg_state) 

    print('deg_manifold_length:', len(deg_manifold))

    ## Calculate the ground state energy
    #w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    #GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    #GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )
    
    # Start from |+> states
    if psi_ref == '+':
        reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )
    
    # Start from symmetry breaking state:
    state_ref = np.full((2**n, 1), 0.0+0.0j)
    # State |0++...>
    if psi_ref == '0':
        for ii in range(2**(n-1)):
            state_ref[ii] = 1/np.sqrt(2**(n-1))
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
        
    # Or state |1++...>
    if psi_ref == '1':
        for ii in range(2**(n-1)):
            state_ref[ii+2**(n-1)] = 1/np.sqrt(2**(n-1))
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
        
    # Or state |(+i)++...>
    if psi_ref == 'i':
        for ii in range(2**(n-1)):
            state_ref[ii] = 1/np.sqrt(2**n)
        for ii in range(2**(n-1)):
            state_ref[ii+2**(n-1)] = 1j/np.sqrt(2**n)
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
    

    reference_bra = reference_ket.transpose().conj()
    E = reference_bra.dot(hamiltonian.dot(reference_ket))[0,0].real

    # Thetas 
    parameters = []
    
    print(" structure :", structure)
    print(" selection :", selection)
    print(" initial parameter:", init_para)
    print(" optimizer:", opt_method)
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []

    min_options = {'gtol': theta_thresh, 'disp':False}
    
    if Tetris == 0:
        f_mix.write("#mixer of individual rotations \n")
    elif Tetris == 1:
        f_mix.write("#mixer of simultaneous rotations \n")

    for p in range(0, layer):
        print(" --------------------------------------------------------------------------")
        print("                                  layer: ", p+1)
        print(" --------------------------------------------------------------------------")
        
        if structure == 'qaoa':
            ansatz_ops.insert(0, pool.cost_ops[0])
            ansatz_mat.insert(0, pool.cost_mat[0])
            parameters.insert(0, init_para)

        if selection == 'NA':
            ansatz_ops.insert(0, pool.mixer_ops[0])
            ansatz_mat.insert(0, pool.mixer_mat[0])
            parameters.insert(0, init_para)
            
            f_mix.write("%d \t %d \n" % (p, 0))
            f_mix.flush()

        if selection == 'grad':
            trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

            curr_state = trial_model.prepare_state(parameters)
            #print("current state: ", curr_state)

            sig = hamiltonian.dot(curr_state)
            
            # Rank the pool operators by gradient
            grad_list = []
            for op_trial in range(pool.n_ops):
                opA = pool.spmat_ops[op_trial]
                com = 2 * (curr_state.transpose().conj().dot(opA.dot(sig))).real
                assert (com.shape == (1, 1))
                com = com[0, 0]
                assert (np.isclose(com.imag, 0))
                com = com.real
                grad_list.append((op_trial, abs(com)))

            grad_list.sort(reverse = True, key = lambda x: x[1])
#            for e in grad_list:
#                print(pool.pauli_ops[e[0]], e[1])

            next_index = grad_list[0][0]
            next_deriv = grad_list[0][1]
            added_op_index = [next_index]
            
            if next_deriv < adapt_thresh:
                f_mix.write("#Ansatz Growth Converged!")
                f_mix.flush()
                break
            
            # Allow different parameters for different operators
            if Tetris == 0:
               #print(" Add operator %4i" % next_index)
                parameters.insert(0,0)
                ansatz_ops.insert(0,pool.pool_ops[next_index])
                ansatz_mat.insert(0,pool.spmat_ops[next_index])
                
                for ii_op in range(1, pool.n_ops):
                    overlap = 0
                    for op_ex in added_op_index:
                        if pool.has_overlap(grad_list[ii_op][0], op_ex):
                            overlap += 1
                            break
                    if overlap == 0:
                        parameters.insert(0,0)
                        ansatz_ops.insert(0,pool.pool_ops[grad_list[ii_op][0]])
                        ansatz_mat.insert(0,pool.spmat_ops[grad_list[ii_op][0]])
                        added_op_index.append(grad_list[ii_op][0])
                        #print(" Add operator %4i" % grad_list[ii_op][0])
                        
            # One parameter for one layer of operators
            elif Tetris == 1:
                composite_mat = pool.spmat_ops[next_index]
                composite_ops = [pool.pool_ops[next_index]]
                
                for ii_op in range(1, pool.n_ops):
                    overlap = 0
                    for op_ex in added_op_index:
                        if pool.has_overlap(grad_list[ii_op][0], op_ex):
                            overlap += 1
                            break
                    if overlap == 0:
                        composite_mat += pool.spmat_ops[grad_list[ii_op][0]]
                        composite_ops.append(pool.pool_ops[grad_list[ii_op][0]])
                        added_op_index.append(grad_list[ii_op][0])
                        #print(" Add operator %4i" % grad_list[ii_op][0])
                
                parameters.insert(0,0)
                ansatz_ops.insert(0,composite_ops)
                ansatz_mat.insert(0,composite_mat)
            
            f_mix.write("%d \t" % (p))
            for el in added_op_index:
                f_mix.write("%d \t" % (el))
            f_mix.write("\n")
            f_mix.flush()
            

            if landscape == True:
                lattice = np.arange(-math.pi, math.pi, 2*math.pi/resolution)
                land = np.zeros(shape=(len(lattice), len(lattice)))
                for i in range(len(lattice)):
                    for j in range(len(lattice)):
                        para_land = parameters.copy()
                        para_land[0] = lattice[i] # gamma, cost parameter
                        # print(para_land)
                        para_land.insert(0, lattice[j])
                        # print(para_land)
                        # print("length of parameters", len(parameters)) 
                        # print("length of para_land", len(para_land))
                        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, para_land)
                        land_state = trial_model.prepare_state(para_land)
                        land[i, j] = trial_model.energy(para_land)
                pickle.dump(land, open('./landscape_%s_%s_%s_%d.p' %(selection, init_para, opt_method, p),'wb'))


        # With the operator for this layer chosen, optimize the parameters
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        if opt_method == 'NM':    
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
                                                         method='Nelder-Mead', callback=None)
#            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
#                                                         method='Nelder-Mead', callback=trial_model.callback)
#            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
#                                                         method='Nelder-Mead', options={'maxiter':4800}, callback=None)
#            minimizer_kwargs = {"method": "Nelder-Mead"}
#            opt_result = scipy.optimize.basinhopping(trial_model.energy, parameters,
#                                                        minimizer_kwargs=minimizer_kwargs, niter=50)
      
        if opt_method == 'BFGS':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                                                            options = min_options, method = 'BFGS', callback=trial_model.callback)
        if opt_method == 'Cobyla':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, 
                                                  method = 'Cobyla')
 
        print(opt_result['message'])
        
        parameters = list(opt_result['x'])
        fin_energy = trial_model.curr_energy.copy()

        # Calculate the entanglement spectrum and entropy
        spec = trial_model.entanglement_spectrum(parameters)
        # Equal bipartition
        entropy = trial_model.entanglement_entropy(parameters)
        # Bipartition 1-(n-1)
        entropy1 = trial_model.entanglement_entropy1(parameters)
        f_ent.write("%d" % (p))
        for e in spec:
            f_ent.write("\t %20.12f" % (e))
        f_ent.write("\t %20.12f" % (entropy))
        f_ent.write("\t %20.12f \n" % (entropy1))
        f_ent.flush()
        
        # Calculate the entanglement spectrum and entropy after projective measurement on qubit 0
        spec = trial_model.entanglement_spectrum_m(parameters)
        # Equal bipartition
        entropy = trial_model.entanglement_entropy_m(parameters)
        # Bipartition 1-(n-1)
        entropy1 = trial_model.entanglement_entropy1_m(parameters)
        f_ent_m.write("%d" % (p))
        for e in spec:
            f_ent_m.write("\t %20.12f" % (e))
        f_ent_m.write("\t %20.12f" % (entropy))
        f_ent_m.write("\t %20.12f \n" % (entropy1))
        f_ent_m.flush()

        print(" Finished: %20.12f" % fin_energy)
        #print(' maxcut objective:', trial_model.curr_energy + pool.shift)
        print(' Error:', trial_model.curr_energy - GS_energy.real)
        print(' Params:', parameters)
        #print('Entanglement_entropy:', entropy)
        sys.stdout.flush()
        
        f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real + pool.shift.real)))
        #f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))
        f.flush()
        
        state_fin = trial_model.prepare_state(parameters)
        overlap = 0.0
        for ii in degenerate_indices:
            overlap += abs(state_fin[ii, 0])**2
        f_Fid.write("%d    %20.15f \n" % (p, overlap))
        f_Fid.flush() 

        #state_fin = trial_model.prepare_state(parameters)
        #(row, col, val) = scipy.sparse.find(state_fin)
        #print('optimal state at layer %d' % (p))
        #for ii in range(len(row)):
        #    print ('%s\t%s' % (row[ii], val[ii]))

    #print("final state: ", trial_model.prepare_state(parameters))
    #state_fin = trial_model.prepare_state(parameters)
    #(row, col, val) = scipy.sparse.find(state_fin)
    #for ii in range(len(row)):
    #    print ('%s\t%s' % (row[ii], val[ii]))
    
    #f_mix.write("# Final ansatz: ")
    #for ind in range(len(ansatz_ops)):
    #    print(" %4s %20f %10s" % (ansatz_ops[ind], parameters[ind], ind))
    #    print("")
    #    if (ind % 2) == 0:
    #        f_mix.write("%s %20f %s & \t" % (ansatz_ops[ind], parameters[ind], ind))
    #    if (ind % 2) == 1:
    #        f_mix.write("%s %20f %s & \t" % ('1j H_C', parameters[ind], ind))
    #f_mix.flush()
    
    for ind in range(layer):
        ansatz_mat_temp = ansatz_mat[2*(layer-ind-1):2*layer]
        param_temp = parameters[2*(layer-ind-1):2*layer]
        model_temp = tUCCSD(hamiltonian, ansatz_mat_temp, reference_ket, param_temp)
        
        # Calculate the entanglement spectrum and entropy
        spec = model_temp.entanglement_spectrum(param_temp)
        # Equal bipartition
        entropy = model_temp.entanglement_entropy(param_temp)
        # Bipartition 1-(n-1)
        entropy1 = model_temp.entanglement_entropy1(param_temp)
        error = (GS_energy.real - model_temp.energy(param_temp)) / GS_energy.real 
        
        f_fin.write("%d" % (ind))
        for e in spec:
            f_fin.write("\t %20.12f" % (e))
        f_fin.write("\t %20.12f" % (entropy))
        f_fin.write("\t %20.12f" % (entropy1))
        f_fin.write("\t %20.12f \n" % (error))
        f_fin.flush()

###########################################################################
# Favor or penalize entangling mixer operators
###########################################################################

def run_pen_mixer(n,
         f,
         f_mix,
         f_ent,
         f_ent_m,
         f_Fid,
         f_fin,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
         layer = 1,
         pool=operator_pools.qaoa(),
         psi_ref = '+',
         init_para = 0.01,
         structure = 'qaoa',
         selection = 'NA',
         penalty = 0,
         rand_ham = 'False',
         opt_method = 'NM',
         landscape = False,
         landscape_after = False,
         resolution = 100
         ):
    # {{{

    hamiltonian = pool.cost_mat[0] * 1j
    # print('hamiltonian:',hamiltonian)

    ## Check about the degeneracy
    tolerance = 1e-12	

    H = np.zeros((2**n, 2**n))
    H = hamiltonian.real
    h = H.diagonal()
    print ('spectrum:', h)
    
    hard_min = np.min(h)
    degenerate_indices = np.argwhere(h < hard_min + tolerance).flatten()
    GS_energy = hard_min

    deg_manifold = []
    for ind in degenerate_indices:
    	deg_state = np.zeros(h.shape)
    	deg_state[ind] = 1.0
    	deg_manifold.append(deg_state) 

    print('deg_manifold_length:', len(deg_manifold))

    ## Calculate the ground state energy
    #w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    #GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    #GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )
    
    # Start from |+> states
    if psi_ref == '+':
        reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )
    
    # Start from symmetry breaking state:
    state_ref = np.full((2**n, 1), 0.0+0.0j)
    # State |0++...>
    if psi_ref == '0':
        for ii in range(2**(n-1)):
            state_ref[ii] = 1/np.sqrt(2**(n-1))
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
        
    # Or state |1++...>
    if psi_ref == '1':
        for ii in range(2**(n-1)):
            state_ref[ii+2**(n-1)] = 1/np.sqrt(2**(n-1))
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
        
    # Or state |(+i)++...>
    if psi_ref == 'i':
        for ii in range(2**(n-1)):
            state_ref[ii] = 1/np.sqrt(2**n)
        for ii in range(2**(n-1)):
            state_ref[ii+2**(n-1)] = 1j/np.sqrt(2**n)
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
    
    reference_bra = reference_ket.transpose().conj()
    E = reference_bra.dot(hamiltonian.dot(reference_ket))[0,0].real

    # Thetas 
    parameters = []
    
    print(" structure :", structure)
    print(" selection :", selection)
    print(" initial parameter:", init_para)
    print(" optimizer:", opt_method)
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []

    min_options = {'gtol': theta_thresh, 'disp':False}

    for p in range(0, layer):
        print(" --------------------------------------------------------------------------")
        print("                                  layer: ", p+1)
        print(" --------------------------------------------------------------------------")
        
        if structure == 'qaoa':
            ansatz_ops.insert(0, pool.cost_ops[0])
            ansatz_mat.insert(0, pool.cost_mat[0])
            parameters.insert(0, init_para)

        if selection == 'NA':
            ansatz_ops.insert(0, pool.mixer_ops[0])
            ansatz_mat.insert(0, pool.mixer_mat[0])
            parameters.insert(0, init_para)
            
            f_mix.write("%d \t %d \n" % (p, 0))
            f_mix.flush()

        if selection == 'grad':
            trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

            curr_state = trial_model.prepare_state(parameters)
            #print("current state: ", curr_state)

            sig = hamiltonian.dot(curr_state)

            next_deriv = 0
            next_index = 0
            # Find the operator in the pool causing the largest descent 
            for op_trial in range(pool.n_ops):

                opA = pool.spmat_ops[op_trial]
                com = 2 * (curr_state.transpose().conj().dot(opA.dot(sig))).real
                assert (com.shape == (1, 1))
                com = com[0, 0]
                assert (np.isclose(com.imag, 0))
                com = com.real

                #print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], com))
                # Subtract from gradient a penalty term if the trial mixer operator is entangling                
                if op_trial > pool.n_single:
                    #compare_term = abs(com) - penalty
                    compare_term = abs(com) * (1 - penalty)
                else:
                    compare_term = abs(com)
   
                if compare_term > abs(next_deriv) + adapt_thresh:
                    next_deriv = compare_term
                    next_index = op_trial

            new_op = pool.pool_ops[next_index]
            new_mat = pool.spmat_ops[next_index]
    
            print(" Add operator %4i" % next_index)
            f_mix.write("%d \t %d \n" % (p, next_index))
            f_mix.flush()

            parameters.insert(0, 0)
            #parameters.insert(0, init_para)
            ansatz_ops.insert(0, new_op)
            ansatz_mat.insert(0, new_mat)

            if landscape == True:
                lattice = np.arange(-math.pi, math.pi, 2*math.pi/resolution)
                land = np.zeros(shape=(len(lattice), len(lattice)))
                for i in range(len(lattice)):
                    for j in range(len(lattice)):
                        para_land = parameters.copy()
                        para_land[0] = lattice[i] # gamma, cost parameter
                        # print(para_land)
                        para_land.insert(0, lattice[j])
                        # print(para_land)
                        # print("length of parameters", len(parameters)) 
                        # print("length of para_land", len(para_land))
                        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, para_land)
                        land_state = trial_model.prepare_state(para_land)
                        land[i, j] = trial_model.energy(para_land)
                pickle.dump(land, open('./landscape_%s_%s_%s_%d.p' %(selection, init_para, opt_method, p),'wb'))


        # With the operator for this layer chosen, optimize the parameters
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        if opt_method == 'NM':    
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
                                                         method='Nelder-Mead', callback=None)
#            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
#                                                         method='Nelder-Mead', callback=trial_model.callback)
#            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
#                                                         method='Nelder-Mead', options={'maxiter':1000}, callback=None)
#            minimizer_kwargs = {"method": "Nelder-Mead"}
#            opt_result = scipy.optimize.basinhopping(trial_model.energy, parameters,
#                                                        minimizer_kwargs=minimizer_kwargs, niter=50)

        print(opt_result['message'])
        
        if opt_method == 'BFGS':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                                                            options = min_options, method = 'BFGS', callback=trial_model.callback)
        if opt_method == 'Cobyla':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, 
                                                  method = 'Cobyla')
        
        parameters = list(opt_result['x'])
        fin_energy = trial_model.curr_energy.copy()

        # Calculate the entanglement spectrum and entropy
        spec = trial_model.entanglement_spectrum(parameters)
        # Equal bipartition
        entropy = trial_model.entanglement_entropy(parameters)
        # Bipartition 1-(n-1)
        entropy1 = trial_model.entanglement_entropy1(parameters)
        f_ent.write("%d" % (p))
        for e in spec:
            f_ent.write("\t %20.12f" % (e))
        f_ent.write("\t %20.12f" % (entropy))
        f_ent.write("\t %20.12f \n" % (entropy1))
        f_ent.flush()
        
        # Calculate the entanglement spectrum and entropy after projective measurement on qubit 0
        spec = trial_model.entanglement_spectrum_m(parameters)
        # Equal bipartition
        entropy = trial_model.entanglement_entropy_m(parameters)
        # Bipartition 1-(n-1)
        entropy1 = trial_model.entanglement_entropy1_m(parameters)
        f_ent_m.write("%d" % (p))
        for e in spec:
            f_ent_m.write("\t %20.12f" % (e))
        f_ent_m.write("\t %20.12f" % (entropy))
        f_ent_m.write("\t %20.12f \n" % (entropy1))
        f_ent_m.flush()

        print(" Finished: %20.12f" % fin_energy)
        #print(' maxcut objective:', trial_model.curr_energy + pool.shift)
        print(' Error:', trial_model.curr_energy - GS_energy.real)
        print(' Params:', parameters)
        #print('Entanglement_entropy:', entropy)
        sys.stdout.flush()
        
        f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real + pool.shift.real)))
        #f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))
        #f_overlap.write("%d    %20.15f \n" % (p, overlap))
        f.flush()       

        state_fin = trial_model.prepare_state(parameters)
        overlap = 0.0
        for ii in degenerate_indices:
            overlap += abs(state_fin[ii, 0])**2
        f_Fid.write("%d    %20.15f \n" % (p, overlap))
        f_Fid.flush()

        #state_fin = trial_model.prepare_state(parameters)
        #(row, col, val) = scipy.sparse.find(state_fin)
        #print('optimal state at layer %d' % (p))
        #for ii in range(len(row)):
        #    print ('%s\t%s' % (row[ii], val[ii]))

    #print("final state: ", trial_model.prepare_state(parameters))
    #state_fin = trial_model.prepare_state(parameters)
    #(row, col, val) = scipy.sparse.find(state_fin)
    #for ii in range(len(row)):
    #    print ('%s\t%s' % (row[ii], val[ii]))
    
    #f_mix.write("# Final ansatz: ")
    #for ind in range(len(ansatz_ops)):
    #    print(" %4s %20f %10s" % (ansatz_ops[ind], parameters[ind], ind))
    #    print("")
    #    if (ind % 2) == 0:
    #        f_mix.write("%s %20f %s & \t" % (ansatz_ops[ind], parameters[ind], ind))
    #    if (ind % 2) == 1:
    #        f_mix.write("%s %20f %s & \t" % ('1j H_C', parameters[ind], ind))
    #f_mix.flush()
    
    for ind in range(layer):
        ansatz_mat_temp = ansatz_mat[2*(layer-ind-1):2*layer]
        param_temp = parameters[2*(layer-ind-1):2*layer]
        model_temp = tUCCSD(hamiltonian, ansatz_mat_temp, reference_ket, param_temp)
        
        # Calculate the entanglement spectrum and entropy
        spec = model_temp.entanglement_spectrum(param_temp)
        # Equal bipartition
        entropy = model_temp.entanglement_entropy(param_temp)
        # Bipartition 1-(n-1)
        entropy1 = model_temp.entanglement_entropy1(param_temp)
        error = (GS_energy.real - model_temp.energy(param_temp)) / GS_energy.real 
        
        f_fin.write("%d" % (ind))
        for e in spec:
            f_fin.write("\t %20.12f" % (e))
        f_fin.write("\t %20.12f" % (entropy))
        f_fin.write("\t %20.12f" % (entropy1))
        f_fin.write("\t %20.12f \n" % (error))
        f_fin.flush()


###########################################################################
# Symmetry-breaking
###########################################################################

def run_sb(n,
         f,
         f_mix,
         f_ent,
         f_Fid,
         f_fin,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
         layer = 1,
         pool=operator_pools.qaoa_sb(),
         psi_ref = '1',
         init_para = 0.01,
         structure = 'qaoa',
         selection = 'NA',
         rand_ham = 'False',
         opt_method = 'NM',
         landscape = False,
         landscape_after = False,
         resolution = 100
         ):
    # {{{

    hamiltonian = pool.cost_mat[0] * 1j
    # print('hamiltonian:',hamiltonian)

    ## Check about the degeneracy
    tolerance = 1e-12	

    H = np.zeros((2**n, 2**n))
    H = hamiltonian.real
    h = H.diagonal()
    print ('spectrum:', h)
    
    hard_min = np.min(h)
    degenerate_indices = np.argwhere(h < hard_min + tolerance).flatten()
    GS_energy = hard_min

    deg_manifold = []
    for ind in degenerate_indices:
    	deg_state = np.zeros(h.shape)
    	deg_state[ind] = 1.0
    	deg_manifold.append(deg_state) 

    print('deg_manifold_length:', len(deg_manifold))

    ## Calculate the ground state energy
    #w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    #GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    #GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )
    
    # Start from |+> states
    if psi_ref == '+':
        reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )
    
    # Start from symmetry breaking state:
    state_ref = np.full((2**n, 1), 0.0+0.0j)
    
    # State |0++...>
    if psi_ref == '0':
        for ii in range(2**(n-1)):
            state_ref[ii] = 1/np.sqrt(2**(n-1))
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
        
    # State |1++...>
    if psi_ref == '1':
        for ii in range(2**(n-1)):
            state_ref[ii+2**(n-1)] = 1/np.sqrt(2**(n-1))
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
        
    # State |(+i)++...>
    if psi_ref == 'i':
        for ii in range(2**(n-1)):
            state_ref[ii] = 1/np.sqrt(2**n)
        for ii in range(2**(n-1)):
            state_ref[ii+2**(n-1)] = 1j/np.sqrt(2**n)
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
    

    reference_bra = reference_ket.transpose().conj()
    E = reference_bra.dot(hamiltonian.dot(reference_ket))[0,0].real

    # Thetas 
    parameters = []
    
    print(" structure :", structure)
    print(" selection :", selection)
    print(" initial parameter:", init_para)
    print(" optimizer:", opt_method)
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []

    min_options = {'gtol': theta_thresh, 'disp':False}

    for p in range(0, layer):
        print(" --------------------------------------------------------------------------")
        print("                                  layer: ", p+1)
        print(" --------------------------------------------------------------------------")
        
        if structure == 'qaoa':
            ansatz_ops.insert(0, pool.cost_ops[0])
            ansatz_mat.insert(0, pool.cost_mat[0])
            parameters.insert(0, init_para)

        if selection == 'NA':
            ansatz_ops.insert(0, pool.mixer_ops[0])
            ansatz_mat.insert(0, pool.mixer_mat[0])
            parameters.insert(0, init_para)
            
            f_mix.write("%d \t %d \n" % (p, 0))
            f_mix.flush()

        if selection == 'grad':
            trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

            curr_state = trial_model.prepare_state(parameters)
            #print("current state: ", curr_state)

            sig = hamiltonian.dot(curr_state)

            next_deriv = 0
            next_index = 0
            # Find the operator in the pool causing the largest descent 
            for op_trial in range(pool.n_ops):

                opA = pool.spmat_ops[op_trial]
                com = 2 * (curr_state.transpose().conj().dot(opA.dot(sig))).real
                assert (com.shape == (1, 1))
                com = com[0, 0]
                assert (np.isclose(com.imag, 0))
                com = com.real

                #print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], com))
    
                if abs(com) > abs(next_deriv) + adapt_thresh:
                    next_deriv = com
                    next_index = op_trial

            new_op = pool.pool_ops[next_index]
            new_mat = pool.spmat_ops[next_index]
    
            print(" Add operator %4i" % next_index)
            f_mix.write("%d \t %d \n" % (p, next_index))
            f_mix.flush()

            parameters.insert(0, 0)
            #parameters.insert(0, init_para)
            ansatz_ops.insert(0, new_op)
            ansatz_mat.insert(0, new_mat)

            if landscape == True:
                lattice = np.arange(-math.pi, math.pi, 2*math.pi/resolution)
                land = np.zeros(shape=(len(lattice), len(lattice)))
                for i in range(len(lattice)):
                    for j in range(len(lattice)):
                        para_land = parameters.copy()
                        para_land[0] = lattice[i] # gamma, cost parameter
                        # print(para_land)
                        para_land.insert(0, lattice[j])
                        # print(para_land)
                        # print("length of parameters", len(parameters)) 
                        # print("length of para_land", len(para_land))
                        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, para_land)
                        land_state = trial_model.prepare_state(para_land)
                        land[i, j] = trial_model.energy(para_land)
                pickle.dump(land, open('./landscape_%s_%s_%s_%d.p' %(selection, init_para, opt_method, p),'wb'))


        # With the operator for this layer chosen, optimize the parameters
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        if opt_method == 'NM':    
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
                                                         method='Nelder-Mead', callback=None)
#            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
#                                                         method='Nelder-Mead', callback=trial_model.callback)
#            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
#                                                         method='Nelder-Mead', options={'maxiter':1000}, callback=None)
#            minimizer_kwargs = {"method": "Nelder-Mead"}
#            opt_result = scipy.optimize.basinhopping(trial_model.energy, parameters,
#                                                        minimizer_kwargs=minimizer_kwargs, niter=50)

        print(opt_result['message'])
        
        if opt_method == 'BFGS':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                                                            options = min_options, method = 'BFGS', callback=trial_model.callback)
        if opt_method == 'Cobyla':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, 
                                                  method = 'Cobyla')
        
        parameters = list(opt_result['x'])
        fin_energy = trial_model.curr_energy.copy()

        # Calculate the entanglement spectrum and entropy
        spec = trial_model.entanglement_spectrum(parameters)
        # Equal bipartition
        entropy = trial_model.entanglement_entropy(parameters)
        # Bipartition 1-(n-1)
        entropy1 = trial_model.entanglement_entropy1(parameters)
        f_ent.write("%d" % (p))
        for e in spec:
            f_ent.write("\t %20.12f" % (e))
        f_ent.write("\t %20.12f" % (entropy))
        f_ent.write("\t %20.12f \n" % (entropy1))
        f_ent.flush()

        print(" Finished: %20.12f" % fin_energy)
        #print(' maxcut objective:', trial_model.curr_energy + pool.shift)
        print(' Error:', trial_model.curr_energy - GS_energy.real)
        print(' Params:', parameters)
        #print('Entanglement_entropy:', entropy)
        sys.stdout.flush()
        
        f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real + pool.shift.real)))
        #f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))
        #f_overlap.write("%d    %20.15f \n" % (p, overlap))
        f.flush()
        
        state_fin = trial_model.prepare_state(parameters)
        overlap = 0.0
        for ii in degenerate_indices:
            overlap += abs(state_fin[ii, 0])**2
        f_Fid.write("%d    %20.15f \n" % (p, overlap))
        f_Fid.flush()

        #state_fin = trial_model.prepare_state(parameters)
        #(row, col, val) = scipy.sparse.find(state_fin)
        #print('optimal state at layer %d' % (p))
        #for ii in range(len(row)):
        #    print ('%s\t%s' % (row[ii], val[ii]))

    #print("final state: ", trial_model.prepare_state(parameters))
    #state_fin = trial_model.prepare_state(parameters)
    #(row, col, val) = scipy.sparse.find(state_fin)
    #for ii in range(len(row)):
    #    print ('%s\t%s' % (row[ii], val[ii]))
    
    #f_mix.write("# Final ansatz: ")
    #for ind in range(len(ansatz_ops)):
    #    print(" %4s %20f %10s" % (ansatz_ops[ind], parameters[ind], ind))
    #    print("")
    #    if (ind % 2) == 0:
    #        f_mix.write("%s %20f %s & \t" % (ansatz_ops[ind], parameters[ind], ind))
    #    if (ind % 2) == 1:
    #        f_mix.write("%s %20f %s & \t" % ('1j H_C', parameters[ind], ind))
    #f_mix.flush()
    
    for ind in range(layer):
        ansatz_mat_temp = ansatz_mat[2*(layer-ind-1):2*layer]
        param_temp = parameters[2*(layer-ind-1):2*layer]
        model_temp = tUCCSD(hamiltonian, ansatz_mat_temp, reference_ket, param_temp)
        
        # Calculate the entanglement spectrum and entropy
        spec = model_temp.entanglement_spectrum(param_temp)
        # Equal bipartition
        entropy = model_temp.entanglement_entropy(param_temp)
        # Bipartition 1-(n-1)
        entropy1 = model_temp.entanglement_entropy1(param_temp)
        error = (GS_energy.real - model_temp.energy(param_temp)) / GS_energy.real 
        
        f_fin.write("%d" % (ind))
        for e in spec:
            f_fin.write("\t %20.12f" % (e))
        f_fin.write("\t %20.12f" % (entropy))
        f_fin.write("\t %20.12f" % (entropy1))
        f_fin.write("\t %20.12f \n" % (error))
        f_fin.flush()

###########################################################################
# Symmetry-breaking. Favor or penalize entangling mixer operators
###########################################################################

def run_pen_sb(n,
         f,
         f_mix,
         f_ent,
         f_fin,
         f_Fid,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
         layer = 1,
         pool=operator_pools.qaoa_sb(),
         psi_ref = '1',
         init_para = 0.01,
         structure = 'qaoa',
         selection = 'NA',
         penalty = 0.,
         rand_ham = 'False',
         opt_method = 'NM',
         landscape = False,
         landscape_after = False,
         resolution = 100
         ):
    # {{{

    hamiltonian = pool.cost_mat[0] * 1j
    # print('hamiltonian:',hamiltonian)

    ## Check about the degeneracy
    tolerance = 1e-12	

    H = np.zeros((2**n, 2**n))
    H = hamiltonian.real
    h = H.diagonal()
    print ('spectrum:', h)
    
    hard_min = np.min(h)
    degenerate_indices = np.argwhere(h < hard_min + tolerance).flatten()
    GS_energy = hard_min

    deg_manifold = []
    for ind in degenerate_indices:
    	deg_state = np.zeros(h.shape)
    	deg_state[ind] = 1.0
    	deg_manifold.append(deg_state) 

    print('deg_manifold_length:', len(deg_manifold))

    ## Calculate the ground state energy
    #w, v = scipy.sparse.linalg.eigsh(hamiltonian, which='SR')
    #GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    #GS_energy = min(w)
    
    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )
    
    # Start from |+> states
    if psi_ref == '+':
        reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )
    
    # Start from symmetry breaking state:
    state_ref = np.full((2**n, 1), 0.0+0.0j)
    
    # State |0++...>
    if psi_ref == '0':
        for ii in range(2**(n-1)):
            state_ref[ii] = 1/np.sqrt(2**(n-1))
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
        
    # State |1++...>
    if psi_ref == '1':
        for ii in range(2**(n-1)):
            state_ref[ii+2**(n-1)] = 1/np.sqrt(2**(n-1))
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
        
    # State |(+i)++...>
    if psi_ref == 'i':
        for ii in range(2**(n-1)):
            state_ref[ii] = 1/np.sqrt(2**n)
        for ii in range(2**(n-1)):
            state_ref[ii+2**(n-1)] = 1j/np.sqrt(2**n)
        reference_ket = scipy.sparse.csc_matrix(
            state_ref
        )
    

    reference_bra = reference_ket.transpose().conj()
    E = reference_bra.dot(hamiltonian.dot(reference_ket))[0,0].real

    # Thetas 
    parameters = []
    
    print(" structure :", structure)
    print(" selection :", selection)
    print(" initial parameter:", init_para)
    print(" optimizer:", opt_method)
    
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []

    min_options = {'gtol': theta_thresh, 'disp':False}

    for p in range(0, layer):
        print(" --------------------------------------------------------------------------")
        print("                                  layer: ", p+1)
        print(" --------------------------------------------------------------------------")
        
        if structure == 'qaoa':
            ansatz_ops.insert(0, pool.cost_ops[0])
            ansatz_mat.insert(0, pool.cost_mat[0])
            parameters.insert(0, init_para)

        if selection == 'NA':
            ansatz_ops.insert(0, pool.mixer_ops[0])
            ansatz_mat.insert(0, pool.mixer_mat[0])
            parameters.insert(0, init_para)
            
            f_mix.write("%d \t %d \n" % (p, 0))
            f_mix.flush()

        if selection == 'grad':
            trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

            curr_state = trial_model.prepare_state(parameters)
            #print("current state: ", curr_state)

            sig = hamiltonian.dot(curr_state)

            next_deriv = 0
            next_index = 0
            # Find the operator in the pool causing the largest descent 
            for op_trial in range(pool.n_ops):

                opA = pool.spmat_ops[op_trial]
                com = 2 * (curr_state.transpose().conj().dot(opA.dot(sig))).real
                assert (com.shape == (1, 1))
                com = com[0, 0]
                assert (np.isclose(com.imag, 0))
                com = com.real

                #print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], com))
                
                # Subtract from gradient a penalty term if the trial mixer operator is entangling                
                if op_trial > pool.n_single:
                    compare_term = abs(com) * (1 - penalty)
                else:
                    compare_term = abs(com)
    
                if compare_term > abs(next_deriv) + adapt_thresh:
                    next_deriv = compare_term
                    next_index = op_trial

            new_op = pool.pool_ops[next_index]
            new_mat = pool.spmat_ops[next_index]
    
            print(" Add operator %4i" % next_index)
            f_mix.write("%d \t %d \n" % (p, next_index))
            f_mix.flush()

            parameters.insert(0, 0)
            #parameters.insert(0, init_para)
            ansatz_ops.insert(0, new_op)
            ansatz_mat.insert(0, new_mat)

            if landscape == True:
                lattice = np.arange(-math.pi, math.pi, 2*math.pi/resolution)
                land = np.zeros(shape=(len(lattice), len(lattice)))
                for i in range(len(lattice)):
                    for j in range(len(lattice)):
                        para_land = parameters.copy()
                        para_land[0] = lattice[i] # gamma, cost parameter
                        # print(para_land)
                        para_land.insert(0, lattice[j])
                        # print(para_land)
                        # print("length of parameters", len(parameters)) 
                        # print("length of para_land", len(para_land))
                        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, para_land)
                        land_state = trial_model.prepare_state(para_land)
                        land[i, j] = trial_model.energy(para_land)
                pickle.dump(land, open('./landscape_%s_%s_%s_%d.p' %(selection, init_para, opt_method, p),'wb'))


        # With the operator for this layer chosen, optimize the parameters
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        if opt_method == 'NM':    
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
                                                         method='Nelder-Mead', callback=None)
#            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
#                                                         method='Nelder-Mead', callback=trial_model.callback)
#            opt_result = scipy.optimize.minimize(trial_model.energy, parameters,
#                                                         method='Nelder-Mead', options={'maxiter':1000}, callback=None)
#            minimizer_kwargs = {"method": "Nelder-Mead"}
#            opt_result = scipy.optimize.basinhopping(trial_model.energy, parameters,
#                                                        minimizer_kwargs=minimizer_kwargs, niter=50)

        print(opt_result['message'])
        
        if opt_method == 'BFGS':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient, 
                                                            options = min_options, method = 'BFGS', callback=trial_model.callback)
        if opt_method == 'Cobyla':
            opt_result = scipy.optimize.minimize(trial_model.energy, parameters, 
                                                  method = 'Cobyla')
        
        parameters = list(opt_result['x'])
        fin_energy = trial_model.curr_energy.copy()

        # Calculate the entanglement spectrum and entropy
        spec = trial_model.entanglement_spectrum(parameters)
        # Equal bipartition
        entropy = trial_model.entanglement_entropy(parameters)
        # Bipartition 1-(n-1)
        entropy1 = trial_model.entanglement_entropy1(parameters)
        f_ent.write("%d" % (p))
        for e in spec:
            f_ent.write("\t %20.12f" % (e))
        f_ent.write("\t %20.12f" % (entropy))
        f_ent.write("\t %20.12f \n" % (entropy1))
        f_ent.flush()

        print(" Finished: %20.12f" % fin_energy)
        #print(' maxcut objective:', trial_model.curr_energy + pool.shift)
        print(' Error:', trial_model.curr_energy - GS_energy.real)
        print(' Params:', parameters)
        #print('Entanglement_entropy:', entropy)
        sys.stdout.flush()
        
        f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real + pool.shift.real)))
        #f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))
        #f_overlap.write("%d    %20.15f \n" % (p, overlap))
        f.flush()
        
        state_fin = trial_model.prepare_state(parameters)
        overlap = 0.0
        for ii in degenerate_indices:
            overlap += abs(state_fin[ii, 0])**2
        f_Fid.write("%d    %20.15f \n" % (p, overlap))
        f_Fid.flush()

        #state_fin = trial_model.prepare_state(parameters)
        #(row, col, val) = scipy.sparse.find(state_fin)
        #print('optimal state at layer %d' % (p))
        #for ii in range(len(row)):
        #    print ('%s\t%s' % (row[ii], val[ii]))

    #print("final state: ", trial_model.prepare_state(parameters))
    #state_fin = trial_model.prepare_state(parameters)
    #(row, col, val) = scipy.sparse.find(state_fin)
    #for ii in range(len(row)):
    #    print ('%s\t%s' % (row[ii], val[ii]))
    
    #f_mix.write("# Final ansatz: ")
    #for ind in range(len(ansatz_ops)):
    #    print(" %4s %20f %10s" % (ansatz_ops[ind], parameters[ind], ind))
    #    print("")
    #    if (ind % 2) == 0:
    #        f_mix.write("%s %20f %s & \t" % (ansatz_ops[ind], parameters[ind], ind))
    #    if (ind % 2) == 1:
    #        f_mix.write("%s %20f %s & \t" % ('1j H_C', parameters[ind], ind))
    #f_mix.flush()
    
    for ind in range(layer):
        ansatz_mat_temp = ansatz_mat[2*(layer-ind-1):2*layer]
        param_temp = parameters[2*(layer-ind-1):2*layer]
        model_temp = tUCCSD(hamiltonian, ansatz_mat_temp, reference_ket, param_temp)
        
        # Calculate the entanglement spectrum and entropy
        spec = model_temp.entanglement_spectrum(param_temp)
        # Equal bipartition
        entropy = model_temp.entanglement_entropy(param_temp)
        # Bipartition 1-(n-1)
        entropy1 = model_temp.entanglement_entropy1(param_temp)
        error = (GS_energy.real - model_temp.energy(param_temp)) / GS_energy.real 
        
        f_fin.write("%d" % (ind))
        for e in spec:
            f_fin.write("\t %20.12f" % (e))
        f_fin.write("\t %20.12f" % (entropy))
        f_fin.write("\t %20.12f" % (entropy1))
        f_fin.write("\t %20.12f \n" % (error))
        f_fin.flush()
        
        
