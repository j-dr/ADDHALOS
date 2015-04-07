from __future__ import print_function, division
import numpy as np
import os
import struct
import itertools

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
    
    dtype = np.dtype([('id', int), ('delta', np.float64)])
    delta = np.genfromtxt(filepath, dtype=dtype)
    delta = delta[delta['id']!=0]
    #u, uind = np.unique(delta['id'], return_index=True)
    #delta = delta[uind]
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
    print(len(halos[halos['id']==0]))
    halos = halos[halos['id']!=0]
    #u, uind = np.unique(halos['id'], return_index=True)
    #halos = halos[uind]

    return halos
    
    
def readData(indict):
    """
    Takes as input a dictionary with paths to data
    as keys, and the name of the field to select as values.
    Returns a record array with the field names as keys populated
    by the data in the file located at the path provided
    
    """

    paths = indict.keys()

    if len(paths)==1:
        feats = indict.values()
    else:
        feats = [f for f in itertools.chain(*indict.values())]

    dtype = np.dtype([(f, np.float) for f in feats])
    print(dtype)

    for i, path in enumerate(paths):
        if ('rnn' in path) and ('hlist' in path):
            d = readHaloRnn(path)
        elif ('rnn' in path) and ('snapshot' in path):
            d = readParticleRnn(path)
        elif 'hlist' in path:
            d = readHlist(path)
        else:
            print("""This feature is not currently handled, if you would like to use
                     it, please add a new i/o fuction
                     """)

        if i==0:
            data = np.ndarray(len(d),dtype=dtype)
            
        data[indict[path]] = d[indict[path]]
        
    return data
                  

