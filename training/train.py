from __future__ import print_function, division
import numpy as np
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')

import model
import trainio
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def trainModel(hfeatdata, pfeatdata, preddata, modeltype='HistGauss', cv=None, store=False):
    """
    Given dictionaries with keys as paths, values as features to use,
    read in data and train the model.
    
    """
    
    hfeatures = trainio.readData(hfeatdata)
    pfeatures = trainio.readData(pfeatdata)
    pred = trainio.readData(preddata)

    print(len(hfeatures))
    print(len(pred))
    
    if modeltype=='HistGauss':
        mdl = model.HistGauss(hfeatures, pfeatures, pred, store=store)
    elif modeltype=='RF':
        mdl = model.RF(hfeatures, pfeatures, pred, store=store)
    elif modeltype=='pdfRF':
        mdl = model.pdfRF(hfeatures, pfeatures, pred, store=store)
    else:
        print("""Invalid model specification, please provide a valid model
                 for the modeltype keyword""")

    print('Preprocessing')
    mdl.preprocess()

    #mdl.visDensity()

    print('Training')
    mdl.train(cv=cv)
    
    return mdl



if __name__ == "__main__":    
    mt='RF'
    hfeatpath = '/nfs/slac/g/ki/ki22/cosmo/jderose/halos/calcrnn/output/FLb400/snapdir010/rnn_hlist_10'
    pfeatpath = '/nfs/slac/g/ki/ki22/cosmo/jderose/halos/calcrnn/output/FLb400/snapdir010/parts/*rnn*snapshot_downsample_010.*'
    predpath = '/nfs/slac/g/ki/ki21/cosmo/jderose/halos/rockstar/output/FLb400/hlists/hlist_10.list'
    
    hfeatdata = {hfeatpath:['hdelta']}
    pfeatdata = {pfeatpath:['pdelta']}
    preddata = {predpath:['M200b']}
    
    if mt=='RF':
        #mdl = trainModel(hfeatdata, pfeatdata, preddata, modeltype='RF', cv=3)
        mdl = trainModel(hfeatdata, pfeatdata, preddata, modeltype='pdfRF', store=True)
        with open('train_tests/pdfrf_fine_10.p', 'w') as fp:
            joblib.dump(mdl,fp)

        print('visualizing')
        mdl.visModel()
        plt.savefig('train_tests/rf_hlist_10.png')
    else:
        mdl = trainModel(hfeatdata, pfeatdata, preddata)
        print('visualizing')
        mdl.visModel()
        plt.savefig('train_tests/abgp_hlist_99.png')
        with open('train_tests/abgp_99.p', 'w') as fp:
            joblib.dump(mdl,fp)
    



