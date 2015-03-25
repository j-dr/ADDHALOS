from __future__ import print_function, division
import numpy as np
import os
import struct


def readPartRnn(filepath):
    """ 
    Read binary output from calcRnn code for particles 
    into a numpy array

    """
    
    with open(filepath) as fp:
        bytes = fp.read(4*4)
        head = struct.unpack('iiii', bytes)
        print('Number of particles in {0}: {1}'.format(filepath, head[1]))
        print('Number of nearest neighbors in {0}: {1}'.format(filepath, head[2]))   
        delta = np.fromfile(fp, dtype=np.float)

    return delta


def readHaloRnn(filepath):
    """
    Read text output from calcRnn code for halos
    into numpy array

    """
    
    dtype = np.dtype([('id', int), ('rnn', np.float64)])
    delta = np.genfromtxt(filepath, dtype=dtype)

    return delta
    

def readHlist(filepath):
    """
    Read in a Rockstar hlist
    
    """
    
    #Check to see how many fields in hlist
    with open(filepath, 'r') as fp:
        
        l = fp.readline()
        ls = l.split(' ')
        nfields = len(ls)
        print('Number of fields in hlist {0}: {1}'.format(filepath, nfields))
 
    if nfields == 66:
        dtype = np.dtype([('scale',np.float64),('id',int),('mvir',np.float64),('rvir',np.float64),('rs',np.float64),\
                              ('vrms',np.float64),('vmax',np.float64), ('Rs_Klypin',np.float64),('Mvir_all',np.float64),\
                              ('M200b',np.float64),('M200c',np.float64),('M500c',np.float64),('M2500c',np.float64),\
                              ('Macc',np.float64),('Mpeak',np.float64),('Vacc',np.float64),('Vpeak',np.float64)])
        usecols = [0,1,10,11,12,13,16,34,35,36,37,38,39,56,57,58,59]
    
    elif nfields == 67:
        dtype = np.dtype([('scale',np.float64),('id',int),('mvir',np.float64),('rvir',np.float64),('rs',np.float64),\
                              ('vrms',np.float64),('vmax',np.float64), ('Rs_Klypin',np.float64),('Mvir_all',np.float64),\
                              ('M200b',np.float64),('M200c',np.float64),('M500c',np.float64),('M2500c',np.float64),\
                              ('Macc',np.float64),('Mpeak',np.float64),('Vacc',np.float64),('Vpeak',np.float64)])
        usecols = [0,1,10,11,12,13,16,34,35,36,37,38,39,54,55,56,57]
    
    else:
        print('Unrecognized Hlist format, check file or update readHlist with new format')
        raise Exception
        

    halos = np.genfromtxt(filepath,dtype=dtype,usecols=usecols)

    return halos
    
    
