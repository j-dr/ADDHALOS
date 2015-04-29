from __future__ import print_function, division
import model
import random
import trainio
import cPickle
import argparse
import haloio
from bisect import bisect_left


def addHalos(particles, features, model):
    
    ii = np.ndarray(features.shape[-1])
    halos = np.ndarray(len(particles),dtype=model.pred_dtype)
    count = 0
    for i,p in enumerate(particles):

        #Search for correct bin in feature space
        for j in range(features.shape[-1]):
            ii[j] = bisect_left(model.edges[j], features[i,j])

        draw = random.random()
        if draw<=model.php[ii]:
            halos[count] = model.predict(features[i,:])
            count+=1

    halos = halos[:count]
    return halos
            

def main(configfile):
    
    #Read in parameters from the configuration file
    params = haloio.readConfigFile(configfile)
    
    #Load the particle locations/velocities and feature data
    particles = trainio.readParticles(params.particlepath)
    features = trainio.readData(params.featurepath)

    with open(params.modelpath, 'r') as fp:
        mdl = cPickle.load(fp)

    #Add the halos
    halos = addHalos(particles, features, mdl)

    #Write the halos
    fitsio.write(params.outpath, halos)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file')

    
