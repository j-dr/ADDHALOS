from __future__ import print_function, division
import numpy as np
import os



def readPartRnn(filepath):
    """ 
    Read binary output from calcRnn code for particles 
    into a numpy array

    """
    
    with open(filepath) as fp:
        fp.seek(2*4)
        delta = np.fromfile(fp, dtype=np.float)

    return delta

def readHaloRnn(filepath):
    """
    Read text output from calcRnn code for halos
    into numpy array

    """
    

def readRockstar(filepath):
    """
    Read in rockstar info
    
    """

    return 
    
    
