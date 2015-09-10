from __future__ import print_function
from ..training import trainio
from glob import glob
import numpy as np
import fitsio
import os


class _BaseParticles(object):
    
    def __init__(self, hlistdict, blockdict):
        self._hlist(hlistdict)
        self._blocks(blockdict)

    def _blocks(self, blockdict):
        self.blocks = []
        for key in blockdict:
            self.blocks.append(Block(key, blockdict[key]))

    def _hlist(self, hlistdict):
        key = hlistdict.keys()[0]
        self.hlist = Hlist(key, hlistdict[key])

class _BaseData(object):

    def __init__(self, featuredict):
        self.featuredict = featuredict

    def loadFeatures(self):
        return trainio.readData(self.featuredict)

class Snapshot(_BaseParticles):

    def __init__(self, path, snapdict):
        self.snappath = path
        self.z = snapdict['z']
        super(self.__class__, self).__init__(snapdict['hlist'], snapdict['blocks'])

    def loadTrainingData(self):
        for i, block in enumerate(self.blocks):
            if i==0:
                pfeat = block.loadFeatures()
            else:
                pfeat = np.hstack([pfeat, block.loadFeatures()])

        hfeat = self.hlist.loadFeatures()
        pred = self.hlist.loadPreds()

        return pfeat, hfeat, pred
        
class Lightcone(_BaseParticles):
    
    def __init__(self, hlistdict, blockdict):
        super(self.__class__, self).__init__(hlistdict, blockdict)

class Block(_BaseData):

    def __init__(self, particlepath, blockdict):
        self.particlepath = particlepath
        super(self.__class__, self).__init__(blockdict['features'])

class Hlist(_BaseData):
    
    def __init__(self, hlistpath, hlistdict):
        self.hlistpath = hlistpath
        self.preddict = hlistdict['preds']
        super(self.__class__, self).__init__(hlistdict['features'])

    def loadPreds(self):
        return trainio.readData(self.preddict)
        
class TrainingData:

    def __init__(self, snapdict):
        self._snapshots(snapdict)

    def _snapshots(self, snapdict):
        self.snapshots = []
        for key in snapdict.keys():
            self.snapshots.append(Snapshot(key, snapdict[key]))

    def genTrainingArrays(self):
        
        for i, snap in enumerate(self.snapshots):
            if i==0:
                pf, hf, pr = snap.loadTrainingData()
            else:
                p, h, r = snap.loadTrainingData()
                pf = np.hstack([pf, p])
                del p
                hf = np.hstack([hf, h])
                del h
                pr = np.hstack([pr, r])
                del r

        self.pfeatures = pf
        self.hfeatures = hf
        self.pred = pr

class ValidationData:
    
    def __init__(self, hlistdict, blockdict):
        self.hlist = Hlist(hlistdict)
        self.blocks = self._blocks(blockdict)
        
    def _blocks(self, blockdict):
        self.blocks = []
        for key in blockdict:
            self.blocks.append(Block(key, blockdict[key]))
        

class Config:
    
    def  __init__(self, pdict):
        self.modelparams = {}
        for key in pdict.keys():
            if 'model' in key.lower():
                mp = key.split('model')[-1]
                self.modelparams[mp] = pdict[key]

            else:
                if key=='featurelist':
                    print(pdict[key])
                setattr(self,key,pdict[key])
        
        self.genTrainingData()
        self.genValidationData()

    def genTrainingData(self):
        trainsnaps = {}
        if not hasattr(self, 'trainsnaps'):
            print('[Config] No training data found')
            return

        try:
            with open(self.trainsnaps, 'r') as fp:
                while True:
                    p = fp.readline()
                    p = p.strip()
                    if p=='':
                        break
                    ps = p.split()
                    trainsnaps[ps[0]] = {'z':float(ps[1])}

        except:
            #If can't open try using as globpath
            ts = glob(self.trainsnaps)
            zs = np.genfromtxt(self.trainsnapz, dtype=None, names=['tag','z'])
            ztags = ['0'+str(zt) if zt>9 else '00'+str(zt) for zt in zs['tag']]
            for t in ts:
                st = t.split('/')
                zii = np.array([True if str(zt) in st[-1] else False for zt in ztags])
                if len(zii[zii==True])!=1:
                    raise(ValueError('Number of matching redshifts for {0} != 1'.format(t)))

                trainsnaps[t] = {'z':zs['z'][zii][0]}

        for i, snap in enumerate(trainsnaps.keys()):
            sz = trainsnaps[snap]['z']
            ssnap = snap.split('/')
            ssnum = ssnap[-1].split('_')
            ssnum = ssnum[-1].split('.')[0]
            hlistpath = '{0}/{1}_{2}.list'.format(self.trainhlistpath, self.trainhlistbase, ssnum)
            hlb = hlistpath.split('/')
            hlb[-1] = hlb[-1].split('.')[0]
            trainsnaps[snap]['hlist'] = {hlistpath:{}}
            fpaths = ['{0}/{1}/{2}.{3}'.format(self.trainhfeaturepath, ssnap[-1], hlb[-1], feat)\
                      if feat!='z' else sz for feat in self.featurelist]
            flist = [[f] for f in self.featurelist]
            trainsnaps[snap]['hlist'][hlistpath]['features'] = dict(zip(fpaths, flist))
            trainsnaps[snap]['hlist'][hlistpath]['preds'] = {hlistpath:self.predlist}
    
            blockglob = '{0}/{1}*'.format(snap, self.trainblockbase)
            blocks = glob(blockglob)
            trainsnaps[snap]['blocks'] = {b:{} for b in blocks}

            for block in trainsnaps[snap]['blocks'].keys():
                bs = block.split('/')
                ppr = '/'.join(bs[-2:])
                fpaths = ['{0}/{1}.{2}'.format(self.trainpfeaturepath, ppr, feat)\
                          if feat!='z' else sz for feat in self.featurelist]
                trainsnaps[snap]['blocks'][block]['features'] = dict(zip(fpaths, flist))

        self.trainingData = TrainingData(trainsnaps)

    def genValidationData(self):

        if not hasattr(self, 'validparts'):
            print('[Config] No validation data found')
            return

        validparts = {}
        vp = []
        try:
            with open(self.validparts, 'r') as fp:
                while True:
                    p = fp.readline()
                    p = p.strip()
                    if p=='':
                        break
                    vp.append(p)
        except:
            #If can't open try using as globpath
            vp = glob(self.validparts)

        for parts in vp:
            blockglob = '{0}/{1}*'.format(parts, self.validblockbase)
            blocks = glob(blockglob)
            blocks = {b:{} for b in blocks}

            for block in blocks.keys():
                bs = block.split('/')
                ppr = '/'.join(bs[-2:])
                blocks[block]['features'] = {'{0}/{1}.{2}'.format(self.validpfeaturebase, ppr, feat)\
                                                          :feat for feat in self.featurelist}
            validparts.update(blocks)

        self.validationData = ValidationData(self.validhlist, validparts)
            
def reformat_features(basenames, blockbase, nblockbase, featurebase, feature):

    for base in basenames:
        bs = base.split('/')
        blockglob = '{0}/{1}/{2}*'.format(featurebase, bs[-1], blockbase)
        blocks = glob(blockglob)
        for block in blocks:
            bs = block.split('/')
            bs[-1] = bs[-1].replace(blockbase, nblockbase)
            bs[-1]+='.{0}'.format(feature)
            nblock = '/'.join(bs)
            #print('{0} --> {1}'.format(block, nblock))
            #check to make sure we're not writing over anything
            if os.path.exists(nblock):
                print('{0} already exists! Skipping...')
                continue
            os.rename(block, nblock)
    
def readConfigFile(fname):
    
    with open(fname, 'r') as fp:
        linecount = 0
        l = fp.readline()
        keys = []
        values = []

        while l!='':
            l = l.strip()

            if (len(l)==0) or (l[0]=='#'):
                l = fp.readline()
                continue

            ls = l.split()
            for i, e in enumerate(ls):
                if e[0]=='#':
                    ls = ls[:i]

            if len(ls)==1:
                print(ls)
                print("Only one columns detected in a row in the config file.\n"\
                      "Please check your formatting")
                print("Offending line #{0}: {1}".format(linecount, l))
                raise(IOError)
            elif (ls[0]=='featurelist') or (ls[0]=='predlist'):
                keys.append(ls[0])
                values.append(list(ls[1:]))
                l = fp.readline()
                continue
            elif len(ls)>2: 

                print("More than two columns in the config file.\n"\
                      "Please check your formatting")
                print("Offending line #{0}: {1}".format(linecount, l))
                raise(IOError)
            
            keys.append(ls[0])
            values.append(ls[1])
            l = fp.readline()
            linecount+=1

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

    
