# Project the leftmost qubit to state 0 before calculating entanglement entropy
# Usage: run_tetris_ER.py <graph index>

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
    n = 8
    regular = None # Set to the degree at which each node is connected
                # Set to None to manually specify the desired connectivity.
    prob =  0.25 # Specify the probability of each edge
    Weighted = False # Weighted or unweighted graph
    Discrete = False # Edge weights drawn from

    gseed = 209  # Seed for generating random graphs
    wseed = 1 # Seed for generating random edge weights
    rng = np.random.default_rng(seed=wseed)
    rngG = np.random.default_rng(seed=gseed)

    # number of layers
    p = 2

    field_sb = 0
    q = 0
    init_para = 0.01
    psi_ref = '+'

    opt_method = 'NM'
    Tetris = 1

################################################
    # File names
    direcname = './Tetris1-'

    if Weighted:
        if prob is not None:
            namestring = 'n%d-p%s-ws%d-gs%d_G%d' % (n, prob, wseed, gseed, ind)
        elif regular is not None:
            namestring = 'n%d-D%d-ws%d-gs%d_G%d' % (n, regular, wseed, gseed, ind)
    else:
        if prob is not None:
            namestring = 'n%d-p%s-gs%d_G%d' % (n, prob, gseed, ind)
        elif regular is not None:
            namestring = 'n%d-D%d-gs%d_G%d' % (n, regular, gseed, ind)

    filename = direcname + namestring + '_error.txt'
    filename_1 = direcname + namestring + '_op.txt'
    filename_2 = direcname + namestring + '_ent.txt'
    filename_3 = direcname + namestring + '_entm.txt'
    filename_4 = direcname + namestring + '_Fid.txt'
    filename_5 = direcname + namestring + '_fin.txt'

    f = open(filename, "w")
    f_mix = open(filename_1, "w")
    f_ent = open(filename_2, "w")
    f_ent_m = open(filename_3, "w")
    f_Fid = open(filename_4, "w")
    f_fin = open(filename_5, "w")

    f.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    f_mix.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    f_ent.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    f_ent_m.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    f_Fid.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    f_fin.write("# init_para=%s, discrete=%s\n" % (init_para, Discrete))
    
################################################
    # Discard 
    discard = rngG.integers(low=1, high=10000, size=ind)
       
################################################
    # Generate graph
    gg = int(rngG.integers(low=1, high=10000))
    if prob is not None:
        G=nx.erdos_renyi_graph(n, prob, seed=gg)
    elif regular is not None:
        G=nx.random_regular_graph(regular, n, seed=gg)
    nw = len(G.edges.data()) # Number of edge weights

    if not Weighted:
        weights = np.ones(nw)
    else:
        if Discrete:
            discard = rng.integers(low=1, high=10, size=nw*ind)
            weights = rng.integers(low=1, high=10, size=nw)*0.1
        else:
            discard = rng.random(nw*ind)
            weights = rng.random(nw)
    
    elist = []
    if regular is not None or prob is not None:
        for ct, ii in enumerate(G.edges):
            elist.append((ii[0],ii[1],weights[ct]))
    G.add_weighted_edges_from(elist)
    
    pool=operator_pools.qaoa()
    pool.init(n, G)
    pool.generate_SparseMatrix()

    qaoa_methods.run_tetris(n, 
	             f,
	             f_mix,
	             f_ent,
	             f_ent_m,
                     f_Fid,
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
                     Tetris = Tetris, 
	             rand_ham = 'False', 
	             opt_method = opt_method,
	             landscape = False,
	             landscape_after = False,
	             resolution = 100)

    f.close()
    f_mix.close()
    f_ent.close()
    f_ent_m.close()
    f_Fid.close()
    f_fin.close()


################################################
    
if __name__ == "__main__":
    main()
