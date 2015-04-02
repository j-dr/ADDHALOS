from __future__ import print_function
from numpy import np
from scipy import sp
import munge

class Model(object):

    def __init__(self, features, pred):
        
        self.features = features
        self.pred = pred


    def train(self):
        """
        Fit a predictive model to features and pred

        """
        
        self.model = None


    def preprocess(self):
        """
        Clean the data

        """
        
        

    def predict(self, fvec):
        """
        Use the model to predict values using the provided feature vector

        """
        
        return None
        
class HistPoly(Model):

    def preprocess(self, ranges=None, normed=False):
        """
        Create a histogram from the feature data and data to predict
        
        """
        arrays = [self.features, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        self.counts, self.edges = np.histogramdd(histarray.view((np.float, len(histarray.dtype.names)),\
                                                        range=ranges, normed=normed))
    
    
                                               
    



    
