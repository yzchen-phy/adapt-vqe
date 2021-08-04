from __future__ import print_function
import numpy as np
import scipy
import scipy.linalg
import scipy.io
import copy as cp
import scipy.sparse
import scipy.sparse.linalg
import math
import sys

class Variational_Ansatz:
    """
    Assumes that we have some operator, which will be applied in a specific way G*psi or exp{iG}*psi to create a trial state
    """
    def __init__(self,_H, _G, _ref, _params):
        """
        _H      : sparse matrix
        _G_ops  : list of sparse matrices - each corresponds to a variational parameter
        _ref    : reference state vector
        _params : initialized list of parameters
        """

        self.H = _H
        self.G = _G
        self.ref = cp.deepcopy(_ref)
        self.curr_params = _params 
        self.n_params = len(self.curr_params)
        self.hilb_dim = self.H.shape[0] 
        
        self.iter = 0
        self.energy_per_iteration = []
        self.psi_norm = 1.0
        self.n_procs = 1

    def energy(self,params):
        print(" VIRTUAL Class: please override")
        exit()
    
    def gradient(self,params):
        print(" VIRTUAL Class: please override")
        exit()
    
    def prepare_state(self,params):
        print(" VIRTUAL Class: please override")
        exit()

        
    def callback(self,x):
        try:
            err = np.sqrt(np.vdot(self.der, self.der))
            print(" Iter:%4i Current Energy = %20.16f Gradient Norm %10.1e Gradient Max %10.1e" %(self.iter,
                self.curr_energy.real, err, np.max(np.abs(self.der))))
        except:
            print(" Iter:%4i Current Energy = %20.16f Psi Norm Error %10.1e" %(self.iter,
                self.curr_energy.real, 1-self.psi_norm))
        self.iter += 1
        self.energy_per_iteration.append(self.curr_energy)
        sys.stdout.flush()




class tUCCSD(Variational_Ansatz):
    
    def energy(self,params):
        new_state = self.prepare_state(params)
        assert(new_state.transpose().conj().dot(new_state).toarray()[0][0]-1<0.0000001)
        energy = new_state.transpose().conj().dot(self.H.dot(new_state))[0,0]
        assert(np.isclose(energy.imag,0))
        self.curr_energy = energy.real
        return energy.real

    def prepare_state(self,parameters):
        """ 
        Prepare state:
        exp{A1}exp{A2}exp{A3}...exp{An}|ref>
        """
        new_state = self.ref * 1.0
        for k in reversed(range(0, len(parameters))):
            new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*self.G[k]), new_state)
        return new_state
    
    
    def gradient(self,parameters):
        """ 
        """
        grad = []
        new_ket = self.prepare_state(parameters)
        new_bra = new_ket.transpose().conj()
        
        hbra = new_bra.dot(self.H)
        term = 0
        ket = cp.deepcopy(new_ket)
        grad = self.Recurse(parameters, grad, hbra, ket, term)
        self.der = grad
        return np.asarray(grad)

    def Recurse(self, parameters, grad, hbra, ket, term):
        if term == 0:
            hbra = hbra
            ket = ket
        else:
            hbra = (scipy.sparse.linalg.expm_multiply(-self.G[term-1]*parameters[term-1], hbra.transpose().conj())).transpose().conj()
            ket = scipy.sparse.linalg.expm_multiply(-self.G[term-1]*parameters[term-1], ket)
        grad.append((2*hbra.dot(self.G[term]).dot(ket).toarray()[0][0].real))
        if term<len(parameters)-1:
            term += 1
            self.Recurse(parameters, grad, hbra, ket, term)
        return np.asarray(grad)

    def hessian(self,parameters):
        """
        Analytic hessian for VQE optimization
        """
        hess = np.zeros((self.n_params, self.n_params))
        
        new_ket = self.prepare_state(parameters)
        bra_j = new_ket.transpose().conj()
        bra_sigma_j = bra_j.dot(self.H)
        op_j_h_j = self.H

        for j in reversed(range(0,len(parameters))):
            op_j_h_i = cp.deepcopy(op_j_h_j)
            ket_i = cp.deepcopy(bra_j).transpose().conj()
            op_j_d_i = scipy.sparse.eye(self.hilb_dim)

            for i in reversed(range(0,j+1)):
                hess[i,j] = 2.0 * (bra_sigma_j.dot(self.G[j]).dot(op_j_d_i).dot(self.G[i]).dot(ket_i).toarray()[0][0].real 
                                             - bra_j.dot(self.G[j]).dot(op_j_h_i).dot(self.G[i]).dot(ket_i).toarray()[0][0].real)
                hess[j,i] = hess[i,j]
                if(i):
                    op_j_h_i = (scipy.sparse.linalg.expm_multiply(-self.G[i]*parameters[i],op_j_h_i.transpose().conj())).transpose().conj()
                    ket_i = scipy.sparse.linalg.expm_multiply(-self.G[i]*parameters[i],ket_i)
                    op_j_d_i = (scipy.sparse.linalg.expm_multiply(-self.G[i]*parameters[i],op_j_d_i.transpose().conj())).transpose().conj()

            if(j):
                bra_j = (scipy.sparse.linalg.expm_multiply(-self.G[j]*parameters[j],bra_j.transpose().conj())).transpose().conj()
                bra_sigma_j = (scipy.sparse.linalg.expm_multiply(-self.G[j]*parameters[j],bra_sigma_j.transpose().conj())).transpose().conj()
                op_j_h_j = scipy.sparse.linalg.expm_multiply(-self.G[j]*parameters[j],(scipy.sparse.linalg.expm_multiply(-self.G[j]*parameters[j],op_j_h_j)).transpose().conj())

        return hess

    def fd_hessian(self, parameters):
        """
        Finite-differences hessian using analytic gradients for VQE optimization
        """
        hess = np.zeros((self.n_params,self.n_params))
        h = 1.0e-6
        for i in range(0,self.n_params):
            #take forward step
            parameters[i] += h
            gplus = self.gradient(parameters)
            #take backwards step
            parameters[i] -= 2.0*h
            gminus = self.gradient(parameters)
            #restore parameters and compute hessian
            parameters[i] += h
            hess[i] = (gplus - gminus)/(2.0*h)
        return hess

    #Functions for single parameter optimization
    def energy_rotosolve(self, theta, index, parameters):
        parameters[index] = theta[0]
        return self.energy(parameters)

    def gradient_rotosolve(self, theta, index, parameters):
        parameters[index] = theta[0]
        return self.gradient(parameters)[index]



class UCC(Variational_Ansatz):
    
    def energy(self,params):
        new_state = self.prepare_state(params)
        assert(new_state.transpose().conj().dot(new_state).toarray()[0][0]-1<0.0000001)
        energy = new_state.transpose().conj().dot(self.H.dot(new_state))[0,0]
        assert(np.isclose(energy.imag,0))
        self.curr_energy = energy.real
        return energy.real

    def prepare_state(self,parameters):
        """ 
        Prepare state:
        exp{A1+A2+A3...+An}|ref>
        """
        generator = scipy.sparse.csc_matrix((self.hilb_dim, self.hilb_dim), dtype = complex)
        new_state = self.ref * 1.0
        for mat_op in range(0,len(self.G)):
            generator = generator+parameters[mat_op]*self.G[mat_op]
        new_state = scipy.sparse.linalg.expm_multiply(generator, self.ref)
        return new_state
    
    

