# Add a small field to break symmetry
# Usage: run_sb.py <graph index>

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
    p = 15

    field_sb = 0.1
    q = 0
    init_para = 0.01
    psi_ref = '1'

    opt_method = 'NM'
    seed = 2133   # random seed
    rng = np.random.default_rng(seed=seed)

################################################
    # File names
    direcname = './'
    name = 'n%d-D%d-state%s-s%d_G%d' % (n, n-1, psi_ref, seed, ind)
    filename = direcname + name + '_error.txt'
    filename_1 = direcname + name + '_op.txt'
    filename_2 = direcname + name + '_ent.txt'
    filename_3 = direcname + name + '_fin.txt'

    f = open(filename, "w")
    f_mix = open(filename_1, "w")
    f_ent = open(filename_2, "w")
    f_fin = open(filename_3, "w")

    f.write("# init_para=%s, field=%s\n" % (init_para, field_sb))
    f_mix.write("# init_para=%s, field=%s\n" % (init_para, field_sb))
    f_ent.write("# init_para=%s, field=%s\n" % (init_para, field_sb))
    f_fin.write("# init_para=%s, field=%s\n" % (init_para, field_sb))
    
################################################
    # Discard the edge weights before the ind-th one 
    for ii in range(ind):
        discard = rng.random(n*(n-1)//2)
        #discard = rng.integers(low=1, high=10, size=n*(n-1)//2)*0.1
            
################################################
    elist = []
    weights = rng.random(n*(n-1)//2)
    for ii in range(n):
        for jj in range(ii):
            elist.append((jj, ii, weights[jj+(ii*(ii-1)//2)]))
            
    g = nx.Graph()
    g.add_nodes_from(np.arange(0, n, 1))
    g.add_weighted_edges_from(elist)
    
    qaoa_methods.run_sb(n, 
	             g, 
	             field_sb,
	             q,
	             f,
	             f_mix,
	             f_ent,
	             f_fin,
	             adapt_thresh=1e-10, 
	             theta_thresh=1e-7,
	             layer=p, 
	             pool=operator_pools.qaoa_sb(), 
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
