import vqe_methods 
import operator_pools

import sys

from joblib import Parallel, delayed

def geom_point(f):
    r = 1.342
    geometry = [('H',   (0,0,-r*f)), 
                ('Be',  (0,0,0)), 
                ('H',   (0,0,r*f))]
    
    
    filename = "beh2_gsd_t1_r%04.3f.out" %f
    sys.stdout = open(filename, 'w')
    vqe_methods.adapt_vqe(geometry,
            pool = operator_pools.spin_complement_GSD(),
            adapt_thresh = 1e-1)
    
    
    filename = "beh2_gsd_t2_r%04.3f.out" %f
    sys.stdout = open(filename, 'w')
    vqe_methods.adapt_vqe(geometry,
            pool = operator_pools.spin_complement_GSD(),
            adapt_thresh = 1e-2)
   

    filename = "beh2_gsd_t3_r%04.3f.out" %f
    sys.stdout = open(filename, 'w')
    vqe_methods.adapt_vqe(geometry,
            pool = operator_pools.spin_complement_GSD(),
            adapt_thresh = 1e-3)


Parallel(n_jobs=20)(delayed(geom_point)(f/10+.5) for f in range(20))
