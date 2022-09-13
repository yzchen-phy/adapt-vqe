import openfermion
import numpy as np
import copy as cp

from openfermion import *

class PauliOperatorPool:
    def __init__(self):
        self.n_orb = 0
        self.n_occ_a = 0
        self.n_occ_b = 0
        self.n_vir_a = 0
        self.n_vir_b = 0

        self.n_spin_orb = 0
        self.gradient_print_thresh = 0

    def init(self,n_orb,
            n_occ_a=None,
            n_occ_b=None,
            n_vir_a=None,
            n_vir_b=None):
        self.n_orb = n_orb
        self.n_spin_orb = 2*self.n_orb

        if n_occ_a!=None and n_occ_b!=None:
            assert(n_occ_a == n_occ_b)
            self.n_occ = n_occ_a
            self.n_occ_a = n_occ_a
            self.n_occ_b = n_occ_b
            self.n_vir = n_vir_a
            self.n_vir_a = n_vir_a
            self.n_vir_b = n_vir_b
        self.n_ops = 0

        self.generate_SQ_Operators()

    def generate_SQ_Operators(self):
        print("Virtual: Reimplement")
        exit()

    def generate_SparseMatrix(self):
        self.spmat_ops = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.pauli_ops:
            self.spmat_ops.append(transforms.get_sparse_operator(op, n_qubits = self.n_spin_orb))
        assert(len(self.spmat_ops) == self.n_ops)
        return

    def compute_gradient_i(self,i,v,sig):
        """
        For a previously optimized state |n>, compute the gradient g(k) of exp(c(k) A(k))|n>
        g(k) = 2Real<HA(k)>
        Note - this assumes A(k) is an antihermitian operator. If this is not the case, the derived class should
        reimplement this function. Of course, also assumes H is hermitian
        v   = current_state
        sig = H*v
        """
        opA = self.spmat_ops[i]
        gi = 2*(sig.transpose().conj().dot(opA.dot(v)))
        assert(gi.shape == (1,1))
        gi = gi[0,0]
        assert(np.isclose(gi.imag,0))
        gi = gi.real

        return gi
        
    def has_overlap(self, j, k):
        """
        check qubit support of operator j and support of operator k
        return True if they have overlap
        """
        # list of qubits in operator j
        op_j = self.pauli_ops[j]
        qubit_list_j = [sing_pauli[0] for sing_pauli in list(op_j.terms.keys())[0]]
        
        # list of qubits in operator k
        op_k = self.pauli_ops[k]
        qubit_list_k = [sing_pauli[0] for sing_pauli in list(op_k.terms.keys())[0]]
        
        ct = 0
        for ele1 in qubit_list_j:
            for ele2 in qubit_list_k:
                if ele1 == ele2:
                    ct +=1
                    break
            if ct > 0:
                break
        
        return ct > 0


class Pauli_qubit(PauliOperatorPool):
# {{{
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        
        conserves S_z mod 2
        """

        print(" Form Pauli operators")

        self.pauli_ops = []
        
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p+1,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                termA1 =  QubitOperator('X%d Y%d' % (pa, qa), -1.0j)
                termA2 =  QubitOperator('Y%d X%d' % (pa, qa), -1.0j)
                termB1 =  QubitOperator('X%d Y%d' % (pb, qb), -1.0j)
                termB2 =  QubitOperator('Y%d X%d' % (pb, qb), -1.0j)

                self.pauli_ops.append(termA1)
                self.pauli_ops.append(termA2)
                self.pauli_ops.append(termB1)
                self.pauli_ops.append(termB2)


        for p in range(0,self.n_spin_orb):
            for q in range(p+1,self.n_spin_orb):
                for r in range(q+1,self.n_spin_orb):
                    for s in range(r+1,self.n_spin_orb):
                    
                        if (p+q+r+s) % 2 == 0:
                            termA1 = QubitOperator('Y%d X%d X%d X%d' % (p, q, r, s), -1.0j)
                            termA2 = QubitOperator('X%d Y%d X%d X%d' % (p, q, r, s), -1.0j)
                            termA3 = QubitOperator('X%d X%d Y%d X%d' % (p, q, r, s), -1.0j)
                            termA4 = QubitOperator('X%d X%d X%d Y%d' % (p, q, r, s), -1.0j)
                            termA5 = QubitOperator('X%d Y%d Y%d Y%d' % (p, q, r, s), -1.0j)
                            termA6 = QubitOperator('Y%d X%d Y%d Y%d' % (p, q, r, s), -1.0j)
                            termA7 = QubitOperator('Y%d Y%d X%d Y%d' % (p, q, r, s), -1.0j)
                            termA8 = QubitOperator('Y%d Y%d Y%d X%d' % (p, q, r, s), -1.0j)
                        
                            self.pauli_ops.append(termA1)
                            self.pauli_ops.append(termA2)
                            self.pauli_ops.append(termA3)
                            self.pauli_ops.append(termA4)
                            self.pauli_ops.append(termA5)
                            self.pauli_ops.append(termA6)
                            self.pauli_ops.append(termA7)
                            self.pauli_ops.append(termA8)

        self.n_ops = len(self.pauli_ops)
        print(" Number of operators: ", self.n_ops)
        
        return
# }}}


class Pauli_QEB(PauliOperatorPool):
# {{{
    def double_exc(self, p1, p2, p3, p4):
        """
        generate the qubit excitation operator
        """
        A = QubitOperator('Y%d X%d X%d X%d' % (p1, p2, p3, p4), -0.125j) 
        A += QubitOperator('X%d Y%d X%d X%d' % (p1, p2, p3, p4), -0.125j) 
        A += QubitOperator('Y%d Y%d X%d Y%d' % (p1, p2, p3, p4), -0.125j) 
        A += QubitOperator('Y%d Y%d Y%d X%d' % (p1, p2, p3, p4), -0.125j) 
        A += QubitOperator('X%d X%d Y%d X%d' % (p1, p2, p3, p4), 0.125j)    # relative negative sign
        A += QubitOperator('X%d X%d X%d Y%d' % (p1, p2, p3, p4), 0.125j)    # relative negative sign
        A += QubitOperator('X%d Y%d Y%d Y%d' % (p1, p2, p3, p4), 0.125j)    # relative negative sign 
        A += QubitOperator('Y%d X%d Y%d Y%d' % (p1, p2, p3, p4), 0.125j)    # relative negative sign
        
        return A 
    
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        
        conserves S_z
        """
        print(" Form Pauli operators")

        self.pauli_ops = []
        
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p+1,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                termA =  QubitOperator('X%d Y%d' % (pa, qa), -0.5j) - QubitOperator('Y%d X%d' % (pa, qa), -0.5j)
                termB =  QubitOperator('X%d Y%d' % (pb, qb), -0.5j) - QubitOperator('Y%d X%d' % (pb, qb), -0.5j)

                self.pauli_ops.append(termA)
                self.pauli_ops.append(termB)

        ct = 0
        for p in range(0,self.n_spin_orb):
            for q in range(p+1,self.n_spin_orb):
                for r in range(q+1,self.n_spin_orb):
                    for s in range(r+1,self.n_spin_orb):
                        
                        if (p % 2 == q % 2) and (p % 2 == r % 2) and (p % 2 == s % 2):
                            term1 = self.double_exc(p, q, r, s)
                            term2 = self.double_exc(p, r, q, s)
                            term3 = self.double_exc(p, s, r, q)
                            
                            self.pauli_ops.append(term1)
                            self.pauli_ops.append(term2)
                            self.pauli_ops.append(term3)
                         
                        elif (p % 2 == q % 2) and (q % 2 != r % 2) and (r % 2 == s % 2):

                            term1 = self.double_exc(p, r, q, s)
                            term2 = self.double_exc(p, s, q, r)
                            
                            self.pauli_ops.append(term1)
                            self.pauli_ops.append(term2)

                           
                        elif (p % 2 != q % 2) and (q % 2 == r % 2) and (p % 2 == s % 2):

                            term1 = self.double_exc(p, q, r, s)
                            term2 = self.double_exc(p, r, q, s)
                            
                            self.pauli_ops.append(term1)
                            self.pauli_ops.append(term2)

                        elif (p % 2 != q % 2) and (p % 2 == r % 2) and (q % 2 == s % 2):
                            term1 = self.double_exc(p, q, r, s)
                            term2 = self.double_exc(p, s, r, q)
                            
                            self.pauli_ops.append(term1)
                            self.pauli_ops.append(term2)


        self.n_ops = len(self.pauli_ops)
        print(" Number of operators: ", self.n_ops)
        
        return
# }}}
