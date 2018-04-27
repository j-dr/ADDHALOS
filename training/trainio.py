from __future__ import print_function, division
from collections import namedtuple
import numpy as np
import os
import struct
import itertools
import collections
from glob import glob
from helpers.SimulationAnalysis import readHlist as readHL

__GadgetHeader_fmt = '6I6dddii6Iiiddddii6Ii'
__finenside = 8192

GadgetHeader = namedtuple('GadgetHeader', \
        'npart mass time redshift flag_sfr flag_feedback npartTotal flag_cooling num_files BoxSize Omega0 OmegaLambda HubbleParam flag_age flag_metals NallHW flag_entr_ics')

def readGadgetSnapshot(filename, read_pos=False, read_vel=False, read_id=False,\
        read_mass=False, print_header=False, single_type=-1, lgadget=False):
    """
    This function reads the Gadget-2 snapshot file.

    Parameters
    ----------
    filename : str
        path to the input file
    read_pos : bool, optional
        Whether to read the positions or not. Default is false.
    read_vel : bool, optional
        Whether to read the velocities or not. Default is false.
    read_id : bool, optional
        Whether to read the particle IDs or not. Default is false.
    read_mass : bool, optional
        Whether to read the masses or not. Default is false.
    print_header : bool, optional
        Whether to print out the header or not. Default is false.
    single_type : int, optional
        Set to -1 (default) to read in all particle types.
        Set to 0--5 to read in only the corresponding particle type.
    lgadget : bool, optional
        Set to True if the particle file comes from l-gadget.
        Default is false.

    Returns
    -------
    ret : tuple
        A tuple of the requested data.
        The first item in the returned tuple is always the header.
        The header is in the GadgetHeader namedtuple format.
    """
    blocks_to_read = (read_pos, read_vel, read_id, read_mass)
    ret = []
    with open(filename, 'rb') as f:
        f.seek(4, 1)
        h = list(struct.unpack(__GadgetHeader_fmt, \
                f.read(struct.calcsize(__GadgetHeader_fmt))))
        if lgadget:
            h[30] = 0
            h[31] = h[18]
            h[18] = 0
            single_type = 1
        h = tuple(h)
        header = GadgetHeader._make((h[0:6],) + (h[6:12],) + h[12:16] \
                + (h[16:22],) + h[22:30] + (h[30:36],) + h[36:])
        if print_header:
            print( header )
        if not any(blocks_to_read):
            return header
        ret.append(header)
        f.seek(256 - struct.calcsize(__GadgetHeader_fmt), 1)
        f.seek(4, 1)
        #
        mass_npart = [0 if m else n for m, n in zip(header.mass, header.npart)]
        if single_type not in range(6):
            single_type = -1
        #
        for i, b in enumerate(blocks_to_read):
            if i < 2:
                fmt = np.dtype(np.float32)
                item_per_part = 3
                npart = header.npart
            elif i==2:
                fmt = np.dtype(np.uint64) if lgadget or any(header.NallHW) \
                        else np.dtype(np.uint32)
                item_per_part = 1
                npart = header.npart
            elif i==3:
                fmt = np.dtype(np.float32)
                if sum(mass_npart) == 0:
                    ret.append(np.array([], fmt))
                    break
                item_per_part = 1
                npart = mass_npart
            size_per_part = item_per_part*fmt.itemsize
            #
            f.seek(4, 1)
            if not b:
                f.seek(sum(npart)*size_per_part, 1)
            else:
                if single_type > -1:
                    f.seek(sum(npart[:single_type])*size_per_part, 1)
                    npart_this = npart[single_type]
                else:
                    npart_this = sum(npart)
                data = np.fromstring(f.read(npart_this*size_per_part), fmt)
                if item_per_part > 1:
                    data.shape = (npart_this, item_per_part)
                ret.append(data)
                if not any(blocks_to_read[i+1:]):
                    break
                if single_type > -1:
                    f.seek(sum(npart[single_type+1:])*size_per_part, 1)
            f.seek(4, 1)
    #
    return tuple(ret)

def readPartRnn(filepath):
    """
    Read binary output from calcRnn code for particles
    into a numpy array

    """

    with open(filepath, 'rb') as fp:
        #read header
        bytes = fp.read(4*5)
        head = struct.unpack('iiiii', bytes)
        #read in densities
        bytes = fp.read()
        print(head)
        delta = struct.unpack('{0}f'.format(head[1]), bytes[:-4])
        dtype = np.dtype([('delta', float)])
        #delta = np.array(delta[:-1])
        delta = np.array(delta)
        delta.dtype = dtype

    return delta


def readHaloRnn(filepath):
    """
    Read text output from calcRnn code for halos
    into numpy array

    """

    dtype = np.dtype([('id', int), ('delta', float)])
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
                              ('vrms',float),('vmax',float), ('Rs_Klypin',float),('PX', float),\
                              ('PY', float), ('PZ', float), ('Mvir_all',float), ('M200b',float),\
                              ('M200c',float),('M500c',float),('M2500c',float),('Macc',float),\
                              ('Mpeak',float),('Vacc',float),('Vpeak',float)])
        usecols = [0,1,10,11,12,13,16,34,17,18,19,35,36,37,38,39,56,57,58,59]

    elif nfields == 67:
        dtype = np.dtype([('scale',float),('id',int),('mvir',float),('rvir',float),('rs',float),\
                              ('vrms',float),('vmax',float), ('Rs_Klypin',float),('PX', float),\
                              ('PY', float), ('PZ', float),('Mvir_all',float),('M200b',float),\
                              ('M200c',float),('M500c',float),('M2500c',float),('Macc',float),\
                              ('Mpeak',float),('Vacc',float),('Vpeak',float)])
        usecols = [0,1,10,11,12,13,16,18,19,20,34,35,36,37,38,39,54,55,56,57]

    else:
        print('Unrecognized Hlist format, check file or update readHlist with new format')
        raise Exception


    halos = np.genfromtxt(filepath,dtype=dtype,usecols=usecols)
    halos = halos[halos['id']!=0]

    return halos

def readGadgetParticles(filepath):
    """
    Read in particles from a gadget output file, and return an array
    containing their ids, positions and velocities
    """

    head, pos, vel, idx = readGadgetSnapshot(filepath, read_pos=True,
                                             read_vel=True,
                                             read_id=True)
    dt = np.dtype([('id',int), \
                   ('x',float), ('y',float), ('z',float),\
                   ('vx',float), ('vy',float), ('vz',float)])
    parts = np.hstack((np.atleast_2d(idx).T,
                        pos, vel))
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
    print(indict)
    data = None
    paths = indict.keys()
    feats = [f for f in flatten(indict.values())]
    dt = np.dtype([(f, float) for f in feats])
    print(dt)
    for i, path in enumerate(paths):
        #Check to see what type of reader we need
        if ('delta' in indict[path]) and ('hlist' in str(path)):
            d = readHaloRnn(path)
        elif 'delta' in indict[path]:
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
        elif 'hlist' in str(path):
            d = readHL(path, fields = indict[path])
        elif 'z' in indict[path]:
            if i==0:
                paths.append(path)
                continue
            d = np.zeros(len(d), dtype=np.dtype([('z',float)]))
            d['z']+=path
        else:
            print("""This feature is not currently handled, if you would like to use
                     it, please add a new i/o fuction
                     """)
            return None

        if data==None:
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
