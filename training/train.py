from __future__ import print_function, division
from ..addhalos import haloio
import numpy as np
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import sys
import model
import trainio



def trainModel(hfeatdata, pfeatdata, preddata, **kwargs):
    """
    Given dictionaries with keys as paths, values as features to use,
    read in data and train the model.
    
    """
    hfeatures = trainio.readData(hfeatdata)
    pfeatures = trainio.readData(pfeatdata)
    pred = trainio.readData(preddata)

    if 'type' in kwargs.keys():
        m = getattr(model, kwargs['type'])
    else:
        #use default GMM model
        m = getattr(model, 'GMM')

    mdl = m(hfeatures, pfeatures, pred, **kwargs)

    print('Preprocessing')
    mdl.preprocess()

    print('Training')
    mdl.train(cv=cv)
    
    return mdl



if __name__ == "__main__":    

    cfg = sys.argv[-1]
    params = haloio.readConfigFile(cfg)
    mdl = trainModel(params.hfeaturedict, params.pfeaturedict, params.preddata, \
                     store=True, **params.modelparams)
    print('visualizing')
    mdl.visModel(fname='train_tests/gmm_C250_99_nc200.png')

