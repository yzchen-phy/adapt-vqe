import numpy as np
import scipy 
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, molden, cc
from pyscf.cc import ccsd
import copy as cp
   
import pyscf
from pyscf import lib

from functools import reduce

import openfermion 


def compute_dp_effective_ham(dp_state, op):
    """
    op is a sum of JW operators
    dp_state is a JW_DirectProductState

    Compute <N|..<B|<A|sum_p o_p |A>|B>...|N>
    where, o_p is some JW operator string
    """
   
    print(" Compute effective Hamiltonian")
    #start by dissecting the operators



    #clustered ops is a list of lists of operators for each cluster


    # the goal is to take each term and separate it into a product of terms:
    #   X4X5Z8Z9Y11 -> (X4X5)(Z9)(Y11)
    #               -> (X{4-2}X{5-2}) (Z{9-6}) (Y{11-10}) etc 
    #
    #   However, each cluster is indexed starting at 0 so that it doesn't blow up
    #       assume all qubits are ordered sequentially in groups

  

    dim_state = 1
    for ci in dp_state.vecs:
        if len(ci.shape) > 1:
            dim_state *= ci.shape[1]

    h_eff = scipy.sparse.csc_matrix((dim_state,dim_state)) 
    clustered_ops = []
    for term in op.terms:
        coeff = op.terms[term]
        cl_terms = []
        #print("\n",term, coeff)
        for ci in range(dp_state.n_clusters):
            cl_terms.append(openfermion.QubitOperator((), 1))
        for pp in term:
            ci =  dp_state.qubit_to_cluster[pp[0]]  #cluster index
            qi =  pp[0]-dp_state.clusters[ci][0]    #qubit index for cluster ci
            pauli = pp[1]                           #cluster index
            #print("a:", pp, ci, qi, pauli)
            
            tmp = openfermion.QubitOperator((qi,pauli))
            cl_terms[ci] *= tmp
       
        term_exval = coeff * scipy.sparse.eye(1)
        # now compute expectation values cluster-wise
        for ci in range(dp_state.n_clusters):
            n_qubits_i = len(dp_state.clusters[ci])
            mat = openfermion.transforms.get_sparse_operator(cl_terms[ci], n_qubits=n_qubits_i)
            v = dp_state.vecs[ci]
        
            term_exval = scipy.sparse.kron(term_exval, v.conj().T.dot(mat.dot(v)) )

        h_eff += term_exval

    return h_eff

def compute_dp_expectation_value(dp_state, op):
    """
    op is a sum of JW operators
    dp_state is a JW_DirectProductState

    Compute <N|..<B|<A|sum_p o_p |A>|B>...|N>
    where, o_p is some JW operator string
    """
   
    #start by dissecting the operators



    #clustered ops is a list of lists of operators for each cluster


    # the goal is to take each term and separate it into a product of terms:
    #   X4X5Z8Z9Y11 -> (X4X5)(Z9)(Y11)
    #               -> (X{4-2}X{5-2}) (Z{9-6}) (Y{11-10}) etc 
    #
    #   However, each cluster is indexed starting at 0 so that it doesn't blow up
    #       assume all qubits are ordered sequentially in groups

   
    exval = 0
    clustered_ops = []
    for term in op.terms:
        coeff = op.terms[term]
        cl_terms = []
        #print("\n",term, coeff)
        for ci in range(dp_state.n_clusters):
            cl_terms.append(openfermion.QubitOperator((), 1))
        for pp in term:
            ci =  dp_state.qubit_to_cluster[pp[0]]  #cluster index
            qi =  pp[0]-dp_state.clusters[ci][0]    #qubit index for cluster ci
            pauli = pp[1]                           #cluster index
            #print("a:", pp, ci, qi, pauli)
            
            tmp = openfermion.QubitOperator((qi,pauli))
            cl_terms[ci] *= tmp
       
        term_exval = coeff*1.0
        # now compute expectation values cluster-wise
        for ci in range(dp_state.n_clusters):
            n_qubits_i = len(dp_state.clusters[ci])
            mat = openfermion.transforms.get_sparse_operator(cl_terms[ci], n_qubits=n_qubits_i)
            v = dp_state.vecs[ci]
            term_exval *= v.conj().T.dot(mat.dot(v))

        exval += term_exval

    assert(np.isclose(exval.imag, 0))
    return exval.real

class JW_ClusterBasis:
    def __init__(self):
        self.n_qubits       = 0
        self.clusters       = []
        self.n_clusters     = 0
        self.vecs           = []    #cluster states
        self.total_dim      = 1     #dimension of full space
        self.cluster_dims   = []    #dimensions of each cluster's hilbert space
        self.qubit_to_cluster = []
        self.cluster_n_qubits = []
        
        #thermal stuff
        self.pops           = []    #list of occupation numbers for each cluster state in thermal state
        self.beta           = 1000

    def init(self, clusters):
        """
        clusters is a list of lists of qubits
        destroys current state
        """
    
        self.vecs = []
        self.total_dim
        self.clusters = cp.deepcopy(clusters)
        self.n_clusters = len(self.clusters)
        self.cluster_dims = []
        self.cluster_n_qubits = []
        for c in self.clusters:
            self.cluster_n_qubits.append(len(c))
            self.vecs.append( np.zeros((2**len(c),1)))
            self.total_dim = self.total_dim*(2**len(c))
            self.n_qubits += len(c)
            self.cluster_dims.append(2**len(c))
            self.pops.append(0)

        self.pops[0] = 1

        self.qubit_to_cluster = [0 for i in range(self.n_qubits)]
        for ci in range(self.n_clusters):
            for qi in self.clusters[ci]:
                self.qubit_to_cluster[qi] = ci
        assert(2**self.n_qubits == self.total_dim)


    def compute_initial_guess_vectors(self,sq_ham):
        self.pops = []
        for ci in range(self.n_clusters):
            nqubits = len(self.clusters[ci])
            #orb_ss = self.clusters[ci] 
            orb_ss = range(int(self.clusters[ci][0]/2),int((self.clusters[ci][-1]+1)/2))
            ss_ham_i = sq_ham.extract_local_hamiltonian(orb_ss)
            fe_ham_i  = ss_ham_i.export_FermionOperator()
            ss_ham_mat_i = openfermion.transforms.get_sparse_operator(fe_ham_i)
            
            [curr_e, curr_v] = scipy.sparse.linalg.eigs(ss_ham_mat_i, 9, which="SR")
            #[curr_e, curr_v] = scipy.linalg.eig(ss_ham_mat_i.todense())
            assert(np.all(np.isclose(curr_e.imag,0)))
            curr_e = curr_e.real
            
            idx = curr_e.argsort()   
            curr_e = curr_e[idx]
            curr_v = curr_v[:,idx]
            
            # number ops
            na = openfermion.FermionOperator()
            nb = openfermion.FermionOperator()
            for p in range(len(orb_ss)):
                na += openfermion.FermionOperator(((2*p,1),(2*p,0)),1)
                nb += openfermion.FermionOperator(((2*p+1,1),(2*p+1,0)),1)
            na = openfermion.transforms.get_sparse_operator(na, n_qubits=nqubits)
            nb = openfermion.transforms.get_sparse_operator(nb, n_qubits=nqubits)

            na = curr_v.conj().T.dot(na.dot(curr_v)).diagonal().real
            nb = curr_v.conj().T.dot(nb.dot(curr_v)).diagonal().real
            
            print(" Sub-Block energy:          %-12.8f" %(curr_e[0].real))
           
            self.vecs[ci] = curr_v
            
            beta = self.beta
            e0 = curr_e[0]
            Z = 0
            pops = []
            print(" %4s  %12s  %12s  %6s  %6s  %6s" %("#", "Energy", "Occupation", "Na", "Nb", "N"))
            for i in range(curr_v.shape[1]):
                ei = curr_e[i]-e0
                Z += np.exp(- beta * (ei))
            for i in range(curr_v.shape[1]):
                ei = curr_e[i]-e0
                occnum = np.exp(- beta * (ei))
                pops.append(occnum/Z)
                print(" %4i  %12.8f  %12.8f  %6.3f  %6.3f  %6.3f" %(i, curr_e[i], occnum/Z, na[i], nb[i], na[i]+nb[i]))
            self.pops.append(pops)

    def form_cluster_densities(self):
        self.rho = []
        for ci in range(self.n_clusters):
            v = self.vecs[ci]
            di = v.dot(np.diag(self.pops[ci])).dot(v.T.conj())
            self.rho.append(di)

    def compute_thermal_cluster_exp_val(self, ham):
        assert(self.rho != None)
        e = 0
        for op in ham.ops:
            term = ham.ops[op]
            for ci in range(self.n_clusters):
                opi = op[ci]
                if len(opi) == 0:
                    continue
                term *= np.trace(np.diag(self.pops[ci]).dot(ham.local_ops_mat[ci][opi]))
                #term *= np.trace(self.rho[ci].dot(ham.local_ops_mat[ci][opi]))
            e += term
        assert(np.isclose(e.imag,0))
        e = e.real
        return e


    def compute_thermal_effective_ham(self, ham, i):
        """
        Ha = tr(H pb x pc x ...)
            = sum (Oa) tr(Ob Pb) tr(Oc Pc) ...

        """
        #assert(self.rho != None)
        e = 0
        hi = scipy.sparse.csc_matrix((self.vecs[i].shape[1],self.vecs[i].shape[1]))
        for op in ham.ops:
            term1 = ham.ops[op]
            term2 = 1
            for ci in range(self.n_clusters):
                if ci == i:
                    continue
                opi = op[ci]
                if len(opi) == 0:
                    continue
                term2 *= np.trace(np.diag(self.pops[ci]).dot(ham.local_ops_mat[ci][opi]))
                #term *= np.trace(self.rho[ci].dot(ham.local_ops_mat[ci][opi]))
            term = term1*term2
            hi += ham.local_ops_mat[i][op[i]]*term
        return hi 


class JW_DirectProductState:
    def __init__(self, basis):
        """
        basis: JW_ClusterBasis object
        defaults to ground state
        """
        self.basis = basis
        self.vecs = []
        self.n_clusters = cp.deepcopy(basis.n_clusters)
        for v in basis.vecs:
            self.vecs.append(v[:,0])
        self.cluster_dims = cp.deepcopy(basis.cluster_dims)
        self.qubit_to_cluster = cp.deepcopy(basis.qubit_to_cluster)
        self.clusters = cp.deepcopy(basis.clusters)

    def create_full_vector(self):
        assert(self.n_clusters == len(self.vecs))
        v = cp.deepcopy(self.vecs[0])
        for vi in range(1,self.n_clusters):
            #v = np.tensordot(v,self.vecs[vi],axes=(0,1))
            v = np.kron(v,self.vecs[vi]) 
        assert(v.size == self.basis.total_dim)
        return v



class SQ_Hamiltonian:
    """
    General hamiltonian operator: 
    H = H(pq){p'q} + V(pqrs){p'q'sr} + A(p){p'} + B(p){p} + C(pq){p'q'} + D(pq){pq} + ...
    """
    def __init__(self):
        # operator integrals tensors
        self.int_H = np.array(())
        self.int_V = np.array(())
        self.int_A = np.array(())
        self.int_B = np.array(())
        self.int_C = np.array(())
        self.int_D = np.array(())

        # MO basis : really, just any transformation from AO->Current basis such that C'SC = I
        self.C = np.array(())
                            
        # AO overlap        
        self.S = np.array(())

        self.n_orb = 0

    def init(self, mol, C):
        # molecule is a pyscf molecule object from gto.Mole()
        T = mol.intor('int1e_kin_sph')
        V = mol.intor('int1e_nuc_sph') 
        
        self.S = mol.intor('int1e_ovlp_sph')
        self.C = cp.deepcopy(C) 
        self.n_orb = self.C.shape[1]
        
        self.int_H = T + V
        self.int_V = mol.intor('int2e_sph')

        #self.int_V = pyscf.ao2mo.restore(1, self.int_V, mol.nao_nr())
        self.transform_orbitals(self.C)

    def transform_orbitals(self, U):
        """
        A(p,q)U(q,s) with s is new basis
        """
        assert(U.shape[0] == self.n_orb)

        new_dim = U.shape[1]

        print(" Rotate orbitals")
        self.int_H = np.einsum("pq,pr->rq",self.int_H,U)
        self.int_H = np.einsum("rq,qs->rs",self.int_H,U)
        self.int_V = np.einsum("pqrs,pl->lqrs",self.int_V,U)
        self.int_V = np.einsum("lqrs,qm->lmrs",self.int_V,U)
        self.int_V = np.einsum("lmrs,rn->lmns",self.int_V,U)
        self.int_V = np.einsum("lmns,so->lmno",self.int_V,U)

        if self.int_A.size > 0: 
            self.int_A = np.einsum(' r,rs->s', self.int_A,U) 
        if self.int_B.size > 0: 
            self.int_B = np.einsum(' r,rs->s', self.int_B,U) 
        if self.int_C.size > 0: 
            self.int_C = np.einsum(' pq,qr,rs->ps', U.T,self.int_C,U) 
        if self.int_D.size > 0: 
            self.int_D = np.einsum(' pq,qr,rs->ps', U.T,self.int_D,U) 

        
        self.C = self.C.dot(U)
        self.n_orb = self.C.shape[1]


    def energy_of_determinant(self, config_a, config_b):
        """ This only returns electronic energy"""
        e1 = 0
        e2 = 0
        for i in config_a:
            e1 += self.int_H[i,i]
        for i in config_b:
            e1 += self.int_H[i,i]
        for i in config_a:
            for j in config_a:
                if i>=j:
                    continue
                e2 += self.int_V[i,i,j,j]
                e2 -= self.int_V[i,j,j,i]
        for i in config_b:             
            for j in config_b:         
                if i>=j:                
                    continue           
                e2 += self.int_V[i,i,j,j]
                e2 -= self.int_V[i,j,j,i]
        for i in config_a:             
            for j in config_b:         
                e2 += self.int_V[i,i,j,j]
        e = e1+e2
        return e


    def extract_local_hamiltonian(self,orb_subset):
        """ Extract local Hamiltonian acting only on subset of orbitals """
        assert(len(orb_subset) <= self.n_orb)
        
        H = SQ_Hamiltonian()
        H.C = self.C[:,orb_subset]
        H.n_orb = H.C.shape[1]
        H.int_H = self.int_H[:,orb_subset][orb_subset]
        H.int_V = self.int_V[:,:,:,orb_subset][:,:,orb_subset][:,orb_subset][orb_subset]

        # Note: this doesn't pass other terms - can't see why it'd need to

        return H

    def export_FermionOperator(self, shift=0):

        """
        We have spatial orbital integrals, so we need to convert back to spin orbitals
        """
        fermi_op = openfermion.FermionOperator()

        #H
        for p in range(self.int_H.shape[0]):
            pa = 2*p + shift
            pb = 2*p+1 +  shift
            for q in range(self.int_H.shape[1]):
                qa = 2*q +shift
                qb = 2*q+1 +shift
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,0)), self.int_H[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,0)), self.int_H[p,q]) 
        
        #V
        for p in range(self.int_V.shape[0]):
            pa = 2*p +shift
            pb = 2*p+1 +shift
            for q in range(self.int_V.shape[1]):
                qa = 2*q +shift
                qb = 2*q+1 +shift
                for r in range(self.int_V.shape[2]):
                    ra = 2*r +shift
                    rb = 2*r+1 +shift
                    for s in range(self.int_V.shape[3]):
                        sa = 2*s +shift
                        sb = 2*s+1 +shift
                        #aa
                        fermi_op += .5* openfermion.FermionOperator(((pa,1),(qa,1),(sa,0),(ra,0)), self.int_V[p,r,q,s]) 
                        #ab
                        fermi_op += .5*openfermion.FermionOperator(((pa,1),(qb,1),(sb,0),(ra,0)), self.int_V[p,r,q,s]) 
                        #ba
                        fermi_op += .5*openfermion.FermionOperator(((pb,1),(qa,1),(sa,0),(rb,0)), self.int_V[p,r,q,s]) 
                        #bb
                        fermi_op += .5*openfermion.FermionOperator(((pb,1),(qb,1),(sb,0),(rb,0)), self.int_V[p,r,q,s]) 

        #A
        for p in range(self.int_A.shape[0]):
            pa = 2*p
            pb = 2*p+1
            fermi_op += openfermion.FermionOperator(((pa,1)), self.int_A[p]) 
            fermi_op += openfermion.FermionOperator(((pb,1)), self.int_A[p]) 
    
        #B
        for p in range(self.int_B.shape[0]):
            pa = 2*p
            pb = 2*p+1
            fermi_op += openfermion.FermionOperator(((pa,0)), self.int_B[p]) 
            fermi_op += openfermion.FermionOperator(((pb,0)), self.int_B[p]) 
        
        #C
        for p in range(self.int_C.shape[0]):
            pa = 2*p
            pb = 2*p+1
            for q in range(self.int_C.shape[1]):
                qa = 2*q
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pa,1),(qa,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pa,1),(qb,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qa,1)), self.int_C[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,1),(qb,1)), self.int_C[p,q]) 
        
        #D
        for p in range(self.int_D.shape[0]):
            pa = 2*p
            pb = 2*p+1
            for q in range(self.int_D.shape[1]):
                qa = 2*q
                qb = 2*q+1
                fermi_op += openfermion.FermionOperator(((pa,0),(qa,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pa,0),(qb,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,0),(qa,0)), self.int_D[p,q]) 
                fermi_op += openfermion.FermionOperator(((pb,0),(qb,0)), self.int_D[p,q]) 
        


        return fermi_op











def init(molecule,charge,spin,basis):
# {{{
    #PYSCF inputs
    print(" ---------------------------------------------------------")
    print("                                                          ")
    print("                      Using Pyscf:")
    print("                                                          ")
    print(" ---------------------------------------------------------")
    print("                                                          ")
    mol = gto.Mole()
    mol.atom = molecule

    # this is needed to prevent openblas - openmp clash for some reason
    # todo: take out
    lib.num_threads(1)

    mol.max_memory = 1000 # MB
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis
    mol.build()

    #orbitals and electrons
    n_orb = mol.nao_nr()
    n_b , n_a = mol.nelec 
    nel = n_a + n_b

    #SCF 
    mf = scf.RHF(mol).run()
    #mf = scf.ROHF(mol).run()
    C = mf.mo_coeff #MO coeffs
    S = mf.get_ovlp()

    
    # dump orbitals for viewing 
    molden.from_mo(mol, 'orbitals_canon.molden', C)

    ##READING INTEGRALS FROM PYSCF
    E_nuc = gto.Mole.energy_nuc(mol)
    T = mol.intor('int1e_kin_sph')
    V = mol.intor('int1e_nuc_sph') 
    hcore = T + V
    S = mol.intor('int1e_ovlp_sph')
    g = mol.intor('int2e_sph')

    print("\nSystem and Method:")
    print(mol.atom)

    print("Basis set                                      :%12s" %(mol.basis))
    print("Number of Orbitals                             :%10i" %(n_orb))
    print("Number of electrons                            :%10i" %(nel))
    print("Nuclear Repulsion                              :%16.10f " %E_nuc)
    print("Electronic SCF energy                          :%16.10f " %(mf.e_tot-E_nuc))
    print("SCF Energy                                     :%16.10f"%(mf.e_tot))


    print(" AO->MO")
    g = np.einsum("pqrs,pl->lqrs",g,C)
    g = np.einsum("lqrs,qm->lmrs",g,C)
    g = np.einsum("lmrs,rn->lmns",g,C)
    g = np.einsum("lmns,so->lmno",g,C)

    h = reduce(np.dot, (C.conj().T, hcore, C))

    
    return(n_orb, n_a, n_b, h, g, mol, E_nuc ,mf.e_tot,C,S)
# }}}





def init_pyscf(molecule,charge,spin,basis,orbitals):
# {{{
    #PYSCF inputs
    print(" ---------------------------------------------------------")
    print("                                                          ")
    print("                      Using Pyscf:")
    print("                                                          ")
    print(" ---------------------------------------------------------")
    print("                                                          ")
    mol = gto.Mole()
    mol.atom = molecule

    # this is needed to prevent openblas - openmp clash for some reason
    # todo: take out
    lib.num_threads(1)

    mol.max_memory = 1000 # MB
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis
    mol.build()

    #orbitals and electrons
    n_orb = mol.nao_nr()
    n_b , n_a = mol.nelec 
    nel = n_a + n_b

    #SCF 
    mf = scf.RHF(mol).run()
    #mf = scf.ROHF(mol).run()
    C = mf.mo_coeff #MO coeffs
    S = mf.get_ovlp()

    print(" Orbs1:")
    print(C)
    Cl = cp.deepcopy(C)
    if orbitals == "boys":
        print("\nUsing Boys localised orbitals:\n")
        cl_o = lo.Boys(mol, mf.mo_coeff[:,:n_a]).kernel(verbose=4)
        cl_v = lo.Boys(mol, mf.mo_coeff[:,n_a:]).kernel(verbose=4)
        Cl = np.column_stack((cl_o,cl_v))

    elif orbitals == "pipek":
        print("\nUsing Pipek-Mezey localised orbitals:\n")
        cl_o = lo.PM(mol, mf.mo_coeff[:,:n_a]).kernel(verbose=4)
        cl_v = lo.PM(mol, mf.mo_coeff[:,n_a:]).kernel(verbose=4)
        Cl = np.column_stack((cl_o,cl_v))
    elif orbitals == "edmiston":
        print("\nUsing Edmiston-Ruedenberg localised orbitals:\n")
        cl_o = lo.ER(mol, mf.mo_coeff[:,:n_a]).kernel(verbose=4)
        cl_v = lo.ER(mol, mf.mo_coeff[:,n_a:]).kernel(verbose=4)
        Cl = np.column_stack((cl_o,cl_v))
#    elif orbitals == "svd":
#        print("\nUsing SVD localised orbitals:\n")
#        cl_o = cp.deepcopy(mf.mo_coeff[:,:n_a])
#        cl_v = cp.deepcopy(mf.mo_coeff[:,n_a:])
#
#        [U,s,V] = np.linalg.svd(cl_o)
#        cl_o = cl_o.dot(V.T)
#        Cl = np.column_stack((cl_o,cl_v))
    elif orbitals == "canonical":
        print("\nUsing Canonical orbitals:\n")
        pass
    else:
        print("Error: Wrong orbital specification:")
        exit()

    
    print(" Overlap:")
    print(C.T.dot(S).dot(Cl))

    # sort by cluster
    blocks = [[0,1,2,3],[4,5,6,7]]
    O = Cl[:,:n_a] 
    V = Cl[:,n_a:] 
    [sorted_order, cluster_sizes] = mulliken_clustering(blocks,mol,O)
    O = O[:,sorted_order]
    [sorted_order, cluster_sizes] = mulliken_clustering(blocks,mol,V)
    V = V[:,sorted_order]
    Cl = np.column_stack((O,V)) 
    
    C = cp.deepcopy(Cl)
    # dump orbitals for viewing 
    molden.from_mo(mol, 'orbitals_canon.molden', C)
    molden.from_mo(mol, 'orbitals_local.molden', Cl)

    ##READING INTEGRALS FROM PYSCF
    E_nu = gto.Mole.energy_nuc(mol)
    T = mol.intor('int1e_kin_sph')
    V = mol.intor('int1e_nuc_sph') 
    hcore = T + V
    S = mol.intor('int1e_ovlp_sph')
    g = mol.intor('int2e_sph')

    print("\nSystem and Method:")
    print(mol.atom)

    print("Basis set                                      :%12s" %(mol.basis))
    print("Number of Orbitals                             :%10i" %(n_orb))
    print("Number of electrons                            :%10i" %(nel))
    print("Nuclear Repulsion                              :%16.10f " %E_nu)
    print("Electronic SCF energy                          :%16.10f " %(mf.e_tot-E_nu))
    print("SCF Energy                                     :%16.10f"%(mf.e_tot))


    print(" AO->MO")
    g = np.einsum("pqrs,pl->lqrs",g,C)
    g = np.einsum("lqrs,qm->lmrs",g,C)
    g = np.einsum("lmrs,rn->lmns",g,C)
    g = np.einsum("lmns,so->lmno",g,C)

    h = reduce(np.dot, (C.conj().T, hcore, C))

    
#    #mf = mf.density_fit(auxbasis='weigend')
#    #mf._eri = None
#    mcc = cc.UCCSD(mf)
#    eris = mcc.ao2mo()
#    eris.g = g
#    eris.focka = h
#    eris.fockb = h
#    
#    emp2, t1, t2 = mcc.init_amps(eris)
#    exit()
#    print(abs(t2).sum() - 4.9318753386922278)
#    print(emp2 - -0.20401737899811551)
#    t1, t2 = update_amps(mcc, t1, t2, eris)
#    print(abs(t1).sum() - 0.046961325647584914)
#    print(abs(t2).sum() - 5.378260578551683   )
#    
#    
#    exit()

    return(n_orb, n_a, n_b, h, g, mol, E_nu,mf.e_tot,C,S)
# }}}





def mulliken_clustering(clusters,mol,C):
    aos_per_atom = []
    aos_per_cluster = []

    #check n atoms
    max_atomid = 0
    tmp_natm = 0
    for c in clusters:
        for ci in c:
            max_atomid = max(max_atomid,ci)
            tmp_natm += 1

    assert(max_atomid == tmp_natm-1)
    assert(tmp_natm == mol.natm)
    #for d in dir(mol):
    #    print(d)
    for a in range(mol.natm):
        aos_per_atom.append([])
    for mu in range(C.shape[0]):
        aos_per_atom[mol.bas_atom(mu)].append(mu)

    mos_per_cluster = [[] for i in range(len(clusters))]
    print(" AOs per cluster")
    for c in clusters:
        tmp = []
        for ci in c:
            tmp.extend(aos_per_atom[ci])
        aos_per_cluster.append(tmp)
        print(tmp)

    S = mol.intor('int1e_ovlp_sph')
    for i in range(C.shape[1]):
        print(" Orbital %4i" %i)
        pi = np.einsum('a,b,bc->ac',C[:,i],C[:,i],S)
        
        pop_per_cluster = []

        for c in aos_per_cluster:
            pop_per_cluster.append(np.trace(pi[:,c][c]))
        mos_per_cluster[np.argmax(pop_per_cluster)].append(i)
    print(mos_per_cluster)
    sorted_orb_list = []
    cluster_sizes = []
    for c in mos_per_cluster:
        sorted_orb_list.extend(c)
        cluster_sizes.append(len(c))

    return(sorted_orb_list,cluster_sizes)



    



