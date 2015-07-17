from __future__ import print_function, division
import numpy as np


def expSigmoidWeight(x, a0, a1, xcut):

    return np.exp(-a0/x)/(1+np.exp(-(x-10**xcut)/a1))

def reweight(x, rwfcn):
    """
    Given a list of values, return a subsample of the original sample
    keeping eachvalue with probability rwfcn(x)
    """

    select = np.random.random(len(x))
    rwmask = select <= rwfcn(x)

    return rwmask

def combineRWHalos(ohcat, rhcat, rproxy, fields=['x', 'y', 'z']):
    """
    Given a high resolution halo catalog and a reconstructed low 
    resolution halo catalog, combine them by reweighing each catalog
    based on a function modeling the completeness of the original
    low resolution catalog
    """
    fields.append(rproxy)
    
    hfcn = lambda x : expSigmoidWeight(x, 2e11, 1e10, 12)
    lfcn = lambda x : 1 - expSigmoidWeight(x, 2e11, 1e10, 12)

    hmask = reweight(ohcat[rproxy], hfcn)
    lmask = reweight(rhcat[rproxy], lfcn)

    rwh = ohcat[hmask]
    rwl = rhcat[lmask]

    rwh = rwh[fields]
    rwl = rwl[fields]

    cat = np.hstack([rwh, rwl])

    return cat
    

        
    

    

    

    
