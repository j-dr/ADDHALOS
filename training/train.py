from __future__ import print_function, division
from ..addhalos import haloio
import numpy as np
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import sys
import model
import trainio



def trainModel(config, store=True):
    """
    Given a Config object, load the training data
    train a model on it, saving it to disk
    if desired
    """
    td = config.trainingData
    modelparams = config.modelparams

    if 'type' in modelparams.keys():
        m = getattr(model, modelparams['type'])
    else:
        #use default GMM model
        m = getattr(model, 'GMM')

    td.genTrainingArrays()

    mdl = m(td.hfeatures, td.pfeatures, td.pred, store=store, **modelparams)

    print('Preprocessing')
    mdl.preprocess()

    print('Training')
    mdl.train(cv=cv)
    
    return mdl

if __name__ == "__main__":

    cfg = sys.argv[-1]
    params = haloio.readConfigFile(cfg)
    mdl = trainModel(params, store=True)
    print('visualizing')
    mdl.visModel(fname='train_tests/gmm_C250_99_nc200.png')

