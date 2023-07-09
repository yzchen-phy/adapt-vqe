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

    def entanglement_entropy(self, params):
    
        the_state = self.prepare_state(params)   
        # Bipartition of equal numbers of qubits
        sub_dim = int(np.sqrt(self.hilb_dim))
        density_array = the_state.dot(the_state.conj().transpose()).toarray().reshape((sub_dim, sub_dim) * 2).trace(axis1 = 1, axis2 = 3)
        # density_array = the_state.dot(the_state.conj().transpose()).toarray().reshape((2 ** 3, 2 ** 3) * 2).trace(axis1 = 1, axis2 = 3)
                        
        density = scipy.sparse.csr_matrix(density_array)
        log_pho = scipy.sparse.csr_matrix(scipy.linalg.logm(density_array))

        entropy_0 = -density.dot(log_pho).toarray().trace()

        # if not np.isclose(entropy_0.imag, 0):
        #     print("WARNING: anomalous derivative imaginary part: %12.8f" % entropy_0.imag)

        entropy = entropy_0.real    
        return entropy
        
    def entanglement_entropy1(self, params):
    
        the_state = self.prepare_state(params)    
        # Bipartition of 1 qubit and the rest
        q_num = int(np.log2(self.hilb_dim))
        density_shape = (2, 2)*q_num
        density_array = the_state.dot(the_state.conj().transpose()).toarray().reshape(density_shape)
        entropy_tot = 0.0
        for ii in range(q_num):
            reduced_density = density_array.trace(axis1 = ii, axis2 = ii+q_num)
            reduced_array = reduced_density.reshape(2**(q_num-1), 2**(q_num-1))
            reduced_pho = scipy.sparse.csr_matrix(reduced_array)
            log_pho = scipy.sparse.csr_matrix(scipy.linalg.logm(reduced_array))
            entropy_0 = -reduced_pho.dot(log_pho).toarray().trace()
            entropy_tot +=entropy_0.real
        return entropy_tot
        
    def entanglement_spectrum(self, params):

        the_state = self.prepare_state(params)   
        
        # Bipartition of equal numbers of qubits
        sub_dim = int(np.sqrt(self.hilb_dim))
        density_array = the_state.dot(the_state.conj().transpose()).toarray().reshape((sub_dim, sub_dim) * 2).trace(axis1 = 1, axis2 = 3)
        # log_pho = scipy.linalg.logm(density_array)
        # ent_spec = scipy.linalg.eigvals(log_pho)
        
        evals = scipy.linalg.eigvals(density_array)
        ent_spec = []
        # cutoff value about log2(1e-9)
        x_cutoff = -30
        for e in evals:
            if e.real > 2**x_cutoff:
                ent_spec.append(np.log2(e.real))
            else:
                ent_spec.append(x_cutoff)
        ent_spec.sort()
                        
        return np.asarray(ent_spec)

    def entanglement_entropy_m(self, params):
    
        the_state = self.prepare_state(params)   
        sub_dim = 2 ** (self.q_num//2 - 1)
        # Project the leftmost qubit onto state 0
        slice_state = the_state[0:(self.hilb_dim//2), :]
        projected_state = slice_state / np.sqrt(slice_state.transpose().conj().dot(slice_state).toarray()[0][0])
        assert(projected_state.transpose().conj().dot(projected_state).toarray()[0][0]-1<0.0000001)
        # Bipartition of equal numbers of qubits
        density_array = projected_state.dot(projected_state.conj().transpose()).toarray().reshape((sub_dim, 2*sub_dim) * 2).trace(axis1 = 1, axis2 = 3)
                        
        density = scipy.sparse.csr_matrix(density_array)
        log_pho = scipy.sparse.csr_matrix(scipy.linalg.logm(density_array))
        entropy_0 = -density.dot(log_pho).toarray().trace()

        # if not np.isclose(entropy_0.imag, 0):
        #     print("WARNING: anomalous derivative imaginary part: %12.8f" % entropy_0.imag)

        entropy = entropy_0.real    
        return entropy
        
    def entanglement_entropy1_m(self, params):
    
        the_state = self.prepare_state(params) 
        # Project the leftmost qubit onto state 0
        slice_state = the_state[0:(self.hilb_dim//2), :]
        projected_state = slice_state / np.sqrt(slice_state.transpose().conj().dot(slice_state).toarray()[0][0])
        assert(projected_state.transpose().conj().dot(projected_state).toarray()[0][0]-1<0.0000001)  
        # Bipartition of 1 qubit and the rest
        density_shape = (2, 2)*(self.q_num-1)
        density_array = projected_state.dot(projected_state.conj().transpose()).toarray().reshape(density_shape)
        entropy_tot = 0.0
        for ii in range(self.q_num-1):
            reduced_density = density_array.trace(axis1 = ii, axis2 = ii+self.q_num-1)
            reduced_array = reduced_density.reshape(2**(self.q_num-2), 2**(self.q_num-2))
            reduced_pho = scipy.sparse.csr_matrix(reduced_array)
            log_pho = scipy.sparse.csr_matrix(scipy.linalg.logm(reduced_array))
            entropy_0 = -reduced_pho.dot(log_pho).toarray().trace()
            entropy_tot +=entropy_0.real
        return entropy_tot
        
    def entanglement_spectrum_m(self, params):

        the_state = self.prepare_state(params)   
        sub_dim = 2 ** (self.q_num//2 - 1)
        # Project the leftmost qubit onto state 0
        slice_state = the_state[0:(self.hilb_dim//2), :]
        projected_state = slice_state / np.sqrt(slice_state.transpose().conj().dot(slice_state).toarray()[0][0])
        assert(projected_state.transpose().conj().dot(projected_state).toarray()[0][0]-1<0.0000001)
        # Bipartition of equal numbers of qubits
        density_array =  projected_state.dot(projected_state.conj().transpose()).toarray().reshape((sub_dim, 2*sub_dim) * 2).trace(axis1 = 1, axis2 = 3)
        
        evals = scipy.linalg.eigvals(density_array)
        ent_spec = []
        # cutoff value about log2(1e-9)
        x_cutoff = -30 
        for e in evals:
            if e.real > 2**x_cutoff:
                ent_spec.append(np.log2(e.real))
            else:
                ent_spec.append(x_cutoff)
        ent_spec.sort()
                        
        return np.asarray(ent_spec)
        
 

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
    
    

