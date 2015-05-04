from __future__ import print_function, division
import model
import random
import trainio
import pickle
import argparse
import haloio
import fitsio
import numpy as np
from bisect import bisect_left


def addHalos(particles, features, mdl):

    #Create halo dtype
    n = particles.dtype.names+features.dtype.names+mdl.pred_dtype.names
    v = particles.dtype.fields.values()+features.dtype.fields.values()+mdl.pred_dtype.fields.values()
    t = [sv[0] for sv in v]
    hdt = np.dtype(zip(n,t))
    halos = np.ndarray(len(particles),dtype=hdt)
    hfeat = np.ndarray(len(particles),dtype=features.dtype)
    hpart = np.ndarray(len(particles),dtype=particles.dtype)

    #lists for indexing
    ml = list(mdl.pred_dtype.names)
    fl = list(features.dtype.names)
    pl = list(particles.dtype.names)
    print(ml)
    print(fl)
    print(pl)
    ii = np.ndarray(features.shape[1], dtype=int)
    count = 0
    for i,p in enumerate(particles):
        #Search for correct bin in feature space
        for j in range(len(ii)):
            bi = bisect_left(mdl.edges[j], features[i,j])
            #If higher than largest bin edge, set as last bin
            if bi>(len(mdl.edges[j])-2):
                bi = len(mdl.edges[j])-2

            ii[j] = bi
            
        draw = random.random()
        if draw<=mdl.php[ii]:
            halos[ml][count] = mdl.predict(features[i,:])
            halos[fl][count] = features[i,j]
            halos[pl][count] = p
            count+=1

    halos = halos[:count]
    return halos
            

def main(configfile):
    
    #Read in parameters from the configuration file
    params = haloio.readConfigFile(configfile)

    with open(params.modelpath, 'r') as fp:
        mdl = pickle.load(fp)
        try:
            mdl.reg.n_jobs = -1
        except:
            print('can\'t set n_jobs')
    
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


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file')

    
