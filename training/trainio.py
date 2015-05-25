from __future__ import print_function, division
import numpy as np
import os
import struct
import itertools
import pynbody as pnb
import collections
from glob import glob

def readPartRnn(filepath):
    """ 
    Read binary output from calcRnn code for particles 
    into a numpy array

    """
    
    with open(filepath, 'rb') as fp:
        #read header
        bytes = fp.read(4*5)
        head = struct.unpack('iiiii', bytes)
        print(head)
        #read in densities
        bytes = fp.read()
        delta = struct.unpack('{0}f'.format(head[1]), bytes[:-4])
        dtype = np.dtype([('pdelta', float)])
        #delta = np.array(delta[:-1])
        delta = np.array(delta)
        delta.dtype = dtype

    return delta


def readHaloRnn(filepath):
    """
    Read text output from calcRnn code for halos
    into numpy array

    """
    
    dtype = np.dtype([('id', int), ('hdelta', float)])
    delta = np.genfromtxt(filepath, dtype=dtype)
    delta = delta[delta['id']!=0]
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
        dtype = np.dtype([('scale',float),('id',int),('mvir',float),('rvir',float),('rs',float),\
                              ('vrms',float),('vmax',float), ('Rs_Klypin',float),('Mvir_all',float),\
                              ('M200b',float),('M200c',float),('M500c',float),('M2500c',float),\
                              ('Macc',float),('Mpeak',float),('Vacc',float),('Vpeak',float)])
        usecols = [0,1,10,11,12,13,16,34,35,36,37,38,39,56,57,58,59]
    
    elif nfields == 67:
        dtype = np.dtype([('scale',float),('id',int),('mvir',float),('rvir',float),('rs',float),\
                              ('vrms',float),('vmax',float), ('Rs_Klypin',float),('Mvir_all',float),\
                              ('M200b',float),('M200c',float),('M500c',float),('M2500c',float),\
                              ('Macc',float),('Mpeak',float),('Vacc',float),('Vpeak',float)])
        usecols = [0,1,10,11,12,13,16,34,35,36,37,38,39,54,55,56,57]
    
    else:
        print('Unrecognized Hlist format, check file or update readHlist with new format')
        raise Exception
        

    halos = np.genfromtxt(filepath,dtype=dtype,usecols=usecols)
    print(len(halos[halos['id']==0]))
    halos = halos[halos['id']!=0]

    return halos

def readGadgetParticles(filepath):
    """
    Read in particles from a gadget output file, and return an array
    containing their ids, positions and velocities
    """

    s = pnb.load(filepath)
    dt = np.dtype([('ID',int), \
                   ('PX',float), ('PY',float), ('PZ',float),\
                   ('VX',float), ('VY',float), ('VZ',float)])
    parts = np.hstack((np.atleast_2d(np.array(s['iord'])).T, np.array(s['pos']), np.array(s['vel'])))
    parts.dtype = dt
    return parts

def flatten(l):
    """
    Flatten a nested list
    """
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def readData(indict):
    """
    Takes as input a dictionary with paths to data
    as keys, and the name of the field to select as values.
    Returns a record array with the field names as keys populated
    by the data in the file located at the path provided
    
    """

    paths = indict.keys()
    feats = [f for f in flatten(indict.values())]
    dt = np.dtype([(f, float) for f in feats])

    for i, path in enumerate(paths):
        #Check to see what type of reader we need
        if 'hdelta' in indict[path]:
            d = readHaloRnn(path)
        elif 'pdelta' in indict[path]:
            if '*' in path:
                files = glob(path)
                for j,f in enumerate(files):
                    if j==0:
                        d = readPartRnn(f)
                    else:
                        gd = readPartRnn(f)
                        d = np.hstack((d,gd))
            else:
                d = readPartRnn(path)
        elif 'hlist' in path:
            d = readHlist(path)
        else:
            print("""This feature is not currently handled, if you would like to use
                     it, please add a new i/o fuction
                     """)
            return None

        if i==0:
            data = np.ndarray(len(d),dtype=dt)
            data_view = data.view(float).reshape(len(data), -1)
        
        #Add data from this path to the rec array
        #have to use views to change multiple columns of 
        #rec array
        ii = np.ndarray(len(indict[path]), dtype=int)
        for i in range(len(ii)):
            ii[i] = np.where(np.array(dt.names)==indict[path][i])[0][0]
        
        data_view[:,ii] = d[indict[path]].view(np.float).reshape(len(d),-1)
        
    return data
