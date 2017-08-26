from __future__ import print_function, division
import numpy as np


def join_rec_arrays(arrays):
    """
    Given a list of record arrays of the same length, with unique fields, join them 
    into a single record array
    
    """

    newdtype = sum((a.dtype.descr for a in arrays), [])
    newrecarray = np.empty(len(arrays[0]), dtype = newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]

    return newrecarray
