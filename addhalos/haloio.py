from __future__ import print_function
from glob import glob
import numpy as np
import fitsio


class Snapshot:

    def __init__(self, z, hlistpath, blockdict):
        self.z = z
        self.hlist = Hlist(hlistpath)
        self.blocks = self._blocks(blockdict)

    def _blocks(self, blockdict):
        blocks = []
        for key in blockdict:
            blocks.append(Block(key, blockdict[key]))

class Block:

    def __init__(self, particlepath, featuredict):
        self.particlepath = particlepath
        self.featuredict = featuredict

class Hlist:
    
    def __init__(self, hlistpath):
        self.hlistpath = hlistpath

class TrainingData:

    def __init__(self, snapdict):
        self.snapshots = self._snapshots(snapdict)

    def _snapshots(self, snapdict):
        snapshots = []
        for key in snapdict.keys():
            snapshots.append(key, snapdict[key]['hlist'], snapdict[key]['blocks'])

class ValidationData:
    
    def __init__(self, hlistpath, blockdict):
        self.hlist = Hlist(hlistpath)
        self.blocks = self._blocks(blockdict)
        
    def _blocks(self, blockdict):
        blocks = []
        for key in blockdict:
            blocks.append(Block(key, blockdict[key]))

class Config:
    
    def  __init__(self, pdict):
        self.modelparams = {}
        for key in pdict.keys():
            if 'model' in key.lower():
                mp = key.split('model')[-1]
                self.modelparams[mp] = pdict[key]

            else:
                setattr(self,key,pdict[key])

        self.getParticlePaths()
        self.getFeaturePaths()
        self.getFeaturePaths(halos=True)
        self.getPredPaths()

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

    def getFeaturePaths(self, halos=False):
        """
        Pair feature paths with the feature names to be
        read in from them
        """
        paths = []
        features = []
        if halos!=False:
            zs = []

        if halos==False:
            self.featuredict = {}
            self.pfeaturedict = {}
            path = self.featurepath
        else:
            self.hfeaturedict = {}
            path = self.hfeaturepath
        
        with open(path,'r') as fp:
            if halos==False:
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
                            temp.update({paths[j]:[features[j]]})
                        else:
                            temp[paths[j]].append(features[j])
                        if paths[j] not in self.pfeaturedict.keys():
                            self.pfeaturedict[paths[j]] = [features[j]]
                        else:
                            self.pfeaturedict[paths[j]].append(features[j])

                    self.featuredict.update({self.pplist[i]:temp})
                    paths = []
                    features = []
            else:
                p = fp.readline()
                f = 0
                while p!='':
                    p = p.strip()
                    ps = p.split()
                    if len(ps)!=3:
                        print("Incorrectly formatted feature file!\n "\
                                  "Please make sure number of columns==3\n "\
                                  "and total number of files is equal to\n "\
                                  "Number of features * Number of paths in\n "\
                                  "particlepath")
                        raise
                    zs.append(ps[0])                    
                    paths.append(ps[1])
                    features.append(ps[2])
                    p = fp.readline()
                
                temp = {}
                for j in range(len(paths)):
                    if paths[j] not in temp.keys():
                        temp.update({paths[j]:[features[j]]})
                    else:
                        temp[paths[j]].append(features[j])

                    self.hfeaturedict = temp
                    self.zs = zs

    def getPredPaths(self):
        """
        Pair feature paths with the feature names to be
        read in from them
        """
        paths = []
        features = []
        zs = []
        self.preddict = {}
        path = self.hpredpath
        
        with open(path,'r') as fp:
            p = fp.readline()
            f = 0
            while p!='':
                p = p.strip()
                ps = p.split()
                if len(ps)!=3:
                    print("Incorrectly formatted feature file!\n "\
                              "Please make sure number of columns==3\n "\
                              "and total number of files is equal to\n "\
                              "Number of features * Number of paths in\n "\
                              "particlepath")
                    raise
                zs.append(ps[0])                    
                paths.append(ps[1])
                features.append(ps[2])
                p = fp.readline()
                
            temp = {}
            for j in range(len(paths)):
                if paths[j] not in temp.keys():
                    temp.update({paths[j]:[features[j]]})
                else:
                    temp[paths[j]].append(features[j])
                    
                self.preddict = temp
                if set(zs)!=set(self.zs):
                    raise ValueError("The redshifts in the pred files do not\n "\
                                     "match the redshifts in the halo feature files!\n")


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

    
