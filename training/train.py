from __future__ import print_function, division
import numpy as np
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')

import model
import trainio
import matplotlib.pyplot as plt
import cPickle

def trainModel(featdata, preddata, modeltype='HistGauss', cv=None):
    """
    Given dictionaries with keys as paths, values as features to use,
    read in data and train the model.
    
    """
    
    features = trainio.readData(featdata)
    pred = trainio.readData(preddata)

    print(len(features))
    print(len(pred))
    
    if modeltype=='HistGauss':
        mod = model.HistGauss(features, pred)
    elif modeltype=='RF':
        mod = model.RF(features, pred)
    else:
        print("""Invalid model specification, please provide a valid model
                 for the modeltype keyword""")

    mod.preprocess()
    mod.train(cv=cv)
    
    return mod



if __name__ == "__main__":    
    featpath = '/nfs/slac/g/ki/ki22/cosmo/jderose/halos/calcrnn/output/FLb400/snapdir099/rnn_hlist_99'
    predpath = '/nfs/slac/g/ki/ki21/cosmo/jderose/halos/rockstar/output/FLb400/hlists/hlist_99.list'
    
    featdata = {featpath:'delta'}
    preddata = {predpath:'M200b'}

    mod = trainModel(featdata, preddata)
    mod.vismodel()
    plt.savefig('train_tests/gp_hlist_99.1.png')
    with open('train_tests/gp_99.p', 'w') as fp:
        cpickle.dump(mod,fp)
    



