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

from openfermion import *
from noisyopt import minimizeSPSA

import operator_pools_qaoa
from tVQE import *

############################################################
#    qaoa
############################################################

def run(n,
         g,
         field_sb,
         q,
         f,
         f_mix,
         f_ent,
         f_fin,
         c_ini,
         a_ini,
         gamma = 0.101,
         alpha = 0.602,
         niter_spsa = 200,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
         layer = 1,
         pool=operator_pools.qaoa(),
         psi_ref = '+',
         init_para = 0.01,
         structure = 'qaoa',
         selection = 'NA',
         rand_ham = 'False',
         landscape = False,
         landscape_after = False,
         resolution = 100
         ):
    # {{{

    G = g
    pool.init(n, G, field_sb, q)
    pool.generate_SparseMatrix()

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

    deg_manifold = []
    for ind in degenerate_indices:
    	deg_state = np.zeros(h.shape)
    	deg_state[ind] = 1.0
    	deg_manifold.append(deg_state) 

    print('deg_manifold_length:', len(deg_manifold))

    ## Calculate the ground state energy
    w, v = scipy.sparse.linalg.eigs(hamiltonian, which='SR')
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('energy:', GS_energy.real)
    #print('maxcut objective:', 0.5*GS_energy.real + pool.shift.real )
    
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
    print(" SPSA hyperparameters: c0=", c_ini, ", a0=", a_ini, ", gamma=", gamma, ", alpha=", alpha)
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
            
            # Find the operator in the pool causing the largest descent 
            for op_trial in range(pool.n_ops):

                opA = pool.spmat_ops[op_trial]
                com = 2 * (curr_state.transpose().conj().dot(opA.dot(sig))).real
                assert (com.shape == (1, 1))
                com = com[0, 0]
                assert (np.isclose(com.imag, 0))
                com = com.real

                print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], com))
    
                if abs(com) > abs(next_deriv) + 1e-9:
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
                pickle.dump(land, open('./landscape_%s_g%s_c%s_a%s_%d.p' %(selection, init_para, c_ini, a_ini, p),'wb'))


        # With the operator for this layer chosen, optimize the parameters
        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

############################################################
#    optimize using SPSA
############################################################

        opt_result = minimizeSPSA(
                trial_model.energy,
                parameters,
                niter=niter_spsa, 
                paired=False,
                a=a_ini,
                c=c_ini,
                # alpha=alpha,
                # gamma=gamma,
                callback=None    
                # callback=trial_model.callback
        )
        
        # print(opt_result['message'])
        
############################################################
        
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
        
        f.write("%d   %20.12f   %20.12f\n" % (p, (trial_model.curr_energy - GS_energy.real), (GS_energy.real - trial_model.curr_energy)/(GS_energy.real)))
        #f_overlap.write("%d    %20.15f \n" % (p, overlap))
        f.flush()
        
        state_fin = trial_model.prepare_state(parameters)
        (row, col, val) = scipy.sparse.find(state_fin)
        print('optimal state at layer %d' % (p))
        for ii in range(len(row)):
            print ('%s\t%s' % (row[ii], val[ii]))

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


