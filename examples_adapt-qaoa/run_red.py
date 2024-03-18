# Project the leftmost qubit to state 0 before calculating entanglement entropy
# Usage: run_red.py <graph index>

import sys
sys.path.insert(0, '../src')
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import qaoa_methods as qaoa_methods
import operator_pools_qaoa as operator_pools
import networkx as nx
import numpy as np

def main():
        
    ind = int(sys.argv[1])  # graph index starting from 0
    
    # number of vertices
    n = 6
    # number of layers
    p = 3

    init_para = 0.01
    psi_ref = '+'

    opt_method = 'NM'
    Discrete = True
    seed = 75    # random seed
    rng = np.random.default_rng(seed=seed)

################################################

    # connected pairs in ascending order
    # 6-qubit
    C1 = {(0,1), (1,2), (2,3), (3,4), (4,5)}    # Configuration C1
    #C3 = {(0,1), (1,2), (2,3), (3,4), (4,5), (0,5), (3,6)}     
    conn_pairs = C1

################################################
    # File names
    direcname = './C1-'

    filename = direcname + 'n%d-D%d-s%d_G%d_error' % (n, n-1, seed, ind) + '.txt'
    filename_1 = direcname + 'n%d-D%d-s%d_G%d_op' % (n, n-1, seed, ind) + '.txt'
    filename_2 = direcname + 'n%d-D%d-s%d_G%d_ent' % (n, n-1, seed, ind) + '.txt'
    filename_3 = direcname + 'n%d-D%d-s%d_G%d_entm' % (n, n-1, seed, ind) + '.txt'
    filename_4 = direcname + 'n%d-D%d-s%d_G%d_fin' % (n, n-1, seed, ind) + '.txt'

    f = open(filename, "w")
    f_mix = open(filename_1, "w")
    f_ent = open(filename_2, "w")
    f_ent_m = open(filename_3, "w")
    f_fin = open(filename_4, "w")

    f.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    f_mix.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    f_ent.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    f_ent_m.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    f_fin.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    
################################################
    # Discard 
    for ii in range(ind):
        if Discrete:
            discard = rng.integers(low=1, high=10, size=n*(n-1)//2)*0.1
        else:
            discard = rng.random(n*(n-1)//2)
            
################################################

    elist = []
    if Discrete:
        weights = rng.integers(low=1, high=10, size=n*(n-1)//2)*0.1
    else:
        weights = rng.random(n*(n-1)//2)
    for ii in range(n):
        for jj in range(ii):
            elist.append((jj, ii, weights[jj+(ii*(ii-1)//2)]))
            
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, n, 1))
    g.add_weighted_edges_from(elist)
    
    pool=operator_pools.qaoa_red()
    pool.init(n, g, conn_pairs)
    pool.generate_SparseMatrix()

    qaoa_methods.run(n,
	             f,
	             f_mix,
	             f_ent,
	             f_ent_m,
	             f_fin,
	             adapt_thresh=1e-5, 
	             theta_thresh=1e-7,
	             layer=p, 
	             pool=pool, 
	             psi_ref=psi_ref,
	             init_para=init_para, 
	             structure = 'qaoa', 
	             selection = 'grad', 
#                     selection = 'NA', 
	             rand_ham = 'False', 
	             opt_method = opt_method,
	             landscape = False,
	             landscape_after = False,
	             resolution = 100)

    f.close()
    f_mix.close()
    f_ent.close()
    f_fin.close()

################################################
    
if __name__ == "__main__":
    main()


