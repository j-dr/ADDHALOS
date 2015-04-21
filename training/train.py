from __future__ import print_function, division
import numpy as np
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')

import model
import trainio
import matplotlib.pyplot as plt
import cPickle

def trainModel(hfeatdata, pfeatdata, preddata, modeltype='HistGauss', cv=None):
    """
    Given dictionaries with keys as paths, values as features to use,
    read in data and train the model.
    
    """
    
    hfeatures = trainio.readData(hfeatdata)
    pfeatures = trainio.readData(pfeatdata)
    pred = trainio.readData(preddata)

    print(len(hfeatures))
    #print(len(pfeatures))
    print(len(pred))
    
    if modeltype=='HistGauss':
        mod = model.HistGauss(hfeatures, pfeatures, pred)
    elif modeltype=='RF':
        mod = model.RF(hfeatures, pfeatures, pred)
    else:
        print("""Invalid model specification, please provide a valid model
                 for the modeltype keyword""")

    mod.preprocess()
    mod.train(cv=cv)
    
    return mod



if __name__ == "__main__":    
    mt='GP'
    hfeatpath = '/nfs/slac/g/ki/ki22/cosmo/jderose/halos/calcrnn/output/FLb400/snapdir099/rnn_hlist_99'
    #pfeatpath = '/nfs/slac/g/ki/ki22/cosmo/jderose/halos/calcrnn/output/FLb400/snapdir099/parts/*rnn*snapshot_downsample_099.[0-5]'
    pfeatpath='None'
    predpath = '/nfs/slac/g/ki/ki21/cosmo/jderose/halos/rockstar/output/FLb400/hlists/hlist_99.list'
    

    hfeatdata = {hfeatpath:'delta'}
    pfeatdata = {pfeatpath:'delta'}
    preddata = {predpath:'M200b'}
    
    if mt=='RF':
        mod = trainModel(hfeatdata, pfeatdata, preddata, modeltype='RF')
        mod.vismodel()
        plt.savefig('train_tests/rf_hlist_99.png')
        with open('train_tests/rf_99.p', 'w') as fp:
            cPickle.dump(mod,fp)
    else:
        mod = trainModel(hfeatdata, pfeatdata, preddata)
        mod.vismodel()
        plt.savefig('train_tests/gp_hlist_99.1.png')
        with open('train_tests/gp_99.p', 'w') as fp:
            cPickle.dump(mod,fp)
    



