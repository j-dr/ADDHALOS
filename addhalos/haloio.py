from __future__ import print_function
from glob import glob
import numpy as np
import fitsio


class Config:
    
    def  __init__(self, pdict):

        for key in pdict.keys():
            setattr(self,key,pdict[key])

        self.getParticlePaths()
        self.getFeaturePaths()

    def getParticlePaths(self):
        """
        Read in the paths to the particle files from the 
        file pointed to by particlePath
        """
        self.pplist = []

        with open(self.particlepath,'r') as fp:

            while True:
                p = fp.readline()
                p = p.strip()
                if p=='':
                    break

                self.pplist.append(p)

    def getFeaturePaths(self):
        """
        Pair feature paths with the feature names to be
        read in from them
        """
        paths = []
        features = []
        self.featuredict = {}
        
        with open(self.featurepath,'r') as fp:
            
            for i in range(len(self.pplist)):
                for j in range(len(self.nfeatures)):
                    p = fp.readline()
                    p = p.strip()
                    ps = p.split()
                    if len(ps)!=2:
                        print("Incorrectly formatted feature file!\n "\
                              "Please make sure number of columns==2\n "\
                              "and total number of files is equal to\n "\
                              "Number of features * Number of paths in\n "\
                              "particlepath")
                        raise

                    paths.append(ps[0])
                    features.append(ps[1])
                
                temp = {}
                for j in range(len(paths)):
                    if paths[j] not in temp.keys():
                        print(paths[j])
                        temp.update({paths[j]:[features[j]]})
                    else:
                        temp[paths[j]].append(features[j])

                self.featuredict.update({self.pplist[i]:temp})
                
                paths = []
                features = []



def readConfigFile(fname):
    
    with open(fname, 'r') as fp:
        
        l = fp.readline()
        keys = []
        values = []

        while l!='':
            l = l.strip()

            if (len(l)==0) or (l[0]=='#'): 
                l = fp.readline()
                continue

            ls = l.split()

            if len(ls)==1:
                print(ls)
                print("Only one columns detected in a row in the config file.\n"\
                      "Please check your formatting")
                raise
            elif len(ls)>2: 
                print("More than two columns in the config file.\n"\
                      "Please check your formatting")
                raise
            
            keys.append(ls[0])
            values.append(ls[1])
            l = fp.readline()

    pdict = dict(zip(keys, values))
    return Config(pdict)


def combineHalos(globpath, outpath):
    
    files = glob(globpath)
    fits = fitsio.FITS(outpath, 'rw', vstorage='object')

    for i, f in enumerate(files):
        if i == 0:
            h = fitsio.read(f, ext=1)
        else:
            nh = fitsio.read(f, ext=1)
            h = np.hstack([h,nh])

    fits.update_hdu_list()
    if len(fits.hdu_list)<2:
        fits.write(h)
    else:
        fits[-1].append(h)

    fits.close()

    return h

    
