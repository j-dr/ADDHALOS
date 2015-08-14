from __future__ import print_function, division
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import model
import random
import trainio
import argparse
import haloio
import reweight
import fitsio
import numpy as np
from bisect import bisect_left
from sklearn.externals import joblib
from glob import glob
from mpi4py import MPI


def addHalos(particles, features, mdl):

    #Create halo dtype
    n = particles.dtype.names+features.dtype.names+mdl.pred_dtype.names
    v = particles.dtype.fields.values()+features.dtype.fields.values()+mdl.pred_dtype.fields.values()
    t = [sv[0] for sv in v]
    hdt = np.dtype(zip(n,t))
    halos = np.ndarray(len(particles),dtype=hdt)
    hpred = np.ndarray(len(particles),dtype=mdl.pred_dtype)
    hfeat = np.ndarray(len(particles),dtype=features.dtype)
    hpart = np.ndarray(len(particles),dtype=particles.dtype)

    #lists for indexing
    ml = list(mdl.pred_dtype.names)
    fl = list(features.dtype.names)
    pl = list(particles.dtype.names)
    ii = np.ndarray(features.shape[1], dtype=int)
    count = 0

    for i,p in enumerate(particles):
        if mdl.assignHalo(features[i,:]):
            fvec = features[i,:].view(float).reshape((1,mdl.nfeat))
            hpred[count] = mdl.predict(fvec)
            hfeat[count] = fvec
            hpart[count] = p
            count+=1

    halos['x'] = hpart['x']
    halos['y'] = hpart['y']
    halos['z'] = hpart['z']
    halos['vx'] = hpart['vx']
    halos['vy'] = hpart['vy']
    halos['vz'] = hpart['vz']
    halos['id'] = hpart['id']

    for n in fl:
        halos[n] = hfeat[n]
        
    for n in ml:
        halos[n] = hpred[n]
    
    halos = halos[:count]
    return halos
            

def main(configfile):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    if rank == 0:
        #Read in parameters from the configuration file
        params = haloio.readConfigFile(configfile)
        modelpath = configfile.modelpath
        featuredict = configfile.featuredict
        outpath = configfile.outpath
    else:
        modelpath = None
        featuredict = None
        outpath = None
    
    #send path to model to all processors
    modelpath = comm.broadcast(modelpath, root=0)
    featuredict = comm.broadcast(featuredict, root=0)
    outpath = comm.broadcast(outpath, root=0)

    if rank != 0:
        #if worker, load model
        mdl = joblib.load(modelpath)

    sendcount = 0
    while True:
        if rank == 0:
            if sendcount == len(pplist):
                for i in range(nprocs-1):
                    #if all work done, send terminate signal to workers
                    #and break from loop
                    comm.send(None, dest=i+1, tag=-1)
                break
            else:
                #else wait for ready signal from worker
                comm.recieve(msg, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                comm.send(pplist[sendcount], dest=status.Get_source(), tag=0)
                sendcount+=1
        else:
            #Worker only ever sends ready signals
            comm.send(None, dest=0)
            comm.recieve(pp, source=0, tag=MPI.ANY_TAG, status=status)

            if status.Get_tag == -1:
                #If all work done, break
                break
            else:
                print('[{0}]: Adding halos to {1}'.format(rank, pp))
                particles = trainio.readGadgetParticles(pp)
                features = trainio.readData(featuredict[pp])
                features = np.atleast_2d(features).T
                
                #Add the halos
                halos = addHalos(particles, features, mdl)

                #Write the halos
                pps = pp.split('/')
                op = outpath+'{0}.halo'.format(pps[-1])
                fitsio.write(op, halos)


    if rank == 0:
        #combine halo files into reconstructed halo catalog
        hpath = params.outpath+'*.halo'
        cpath = params.outpath+'/out0.rc.list'
        rhcat = haloio.combineHalos(hpath, cpath)

        #combine original and reconstructed halos
        if 'ohalopath' in params.keys():
            ohcat = trainio.readHL(params.ohalopath, fields=rch.dtype.names)
            h = reweight.combineRWHalos(ohcat, rhcat, params.rproxy)
            fitsio.write(params.outpath+'out0.list')


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file')
    args = parser.parse_args()
    print(args.config)
    main(args.config)
    
