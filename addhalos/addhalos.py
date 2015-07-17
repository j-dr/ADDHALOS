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
        #Search for correct bin in feature space
        for j in range(len(ii)):
            bi = bisect_left(mdl.edges[j], features[i,j][0])
            #If higher than largest bin edge, set as last bin
            if bi>(len(mdl.edges[j])-2):
                bi = len(mdl.edges[j])-2

            ii[j] = bi
    
        #If random draw less than probability of assigning
        #halo to particle with this feature vector then
        #assign halo
        draw = random.random()
        if draw<=mdl.php[ii]:
            hpred[count] = mdl.predict(features[i,:])
            hfeat[count] = features[i,:].item()
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
    
    #Read in parameters from the configuration file
    params = haloio.readConfigFile(configfile)

    mdl = joblib.load(params.modelpath)

    #Load the particle locations/velocities and feature data
    for pp in params.pplist:
        particles = trainio.readGadgetParticles(pp)
        features = trainio.readData(params.featuredict[pp])
        features = np.atleast_2d(features).T

        #Add the halos
        halos = addHalos(particles, features, mdl)

        #Write the halos
        pps = pp.split('/')
        op = params.outpath+'{0}.halo'.format(pps[-1])
        fitsio.write(op, halos)
        
    
    #combine halo files into reconstructed halo catalog
    hpath = params.outpath+'*.halo'
    cpath = params.outpath+'/out0.rc.list'
    rhcat = haloio.combineHalos(hpath, cpath)

    #combine original halos and reconstructed halos
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
    
