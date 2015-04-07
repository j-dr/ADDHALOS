from __future__ import print_function, division
import numpy as np
import model
import trainio


def trainModel(featdata, preddata, modeltype='HistGauss'):
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
    else:
        print("""Invalid model specification, please provide a valid model
                 for the modeltype keyword""")

    mod.preprocess()
    mod.train()
    
    return mod



if __name__ == "__main__":    
    featpath = '/nfs/slac/g/ki/ki22/cosmo/jderose/halos/calcrnn/output/FLb400/snapdir010/rnn_hlist_10'
    predpath = '/nfs/slac/g/ki/ki21/cosmo/jderose/halos/rockstar/output/FLb400/hlists/hlist_10.list'
    
    featdata = {featpath:'delta'}
    preddata = {predpath:'M200b'}

    mod = trainModel(featdata, preddata)




