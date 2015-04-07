from __future__ import print_function
import numpy as np
import scipy as sp
from sklearn import gaussian_process
from abc import ABCMeta, abstractmethod
import seaborn as sns
sns.set_context("talk")
sns.despine(left=True)
import matplotlib.pyplot as plt
import munge

class Model:
    __metaclass__ = ABCMeta

    def __init__(self, features, pred):
        
        self.features = features
        self.pred = pred

    @abstractmethod
    def train(self):
        """
        Fit a predictive model to features and pred

        """
        pass

    @abstractmethod
    def preprocess(self):
        """
        Clean the data

        """
        pass

    @abstractmethod
    def predict(self, fvec):
        """
        Use the model to predict values using the provided feature vector

        """
        pass


        
class HistGauss(Model):


    def preprocess(self):
        """
        Preprocess the data that we will fit the model to

        """
        self.preproc_hist()
        
    
    def preproc_hist(self, ranges=None, nbins=100, normed=False):
        """
        Create a histogram from the feature data and data to predict
        
        """
        arrays = [self.features, self.pred]
        histarray = munge.join_rec_arrays(arrays)

        counts, edges = np.histogramdd(histarray.view((np.float, len(histarray.dtype.names))),\
                                                        range=ranges, bins=nbins, normed=normed)
        centers = [[(edges[i][j]+edges[i][j+1])/2 for j in range(len(edges[i])-1)] for i in range(len(edges))]
        self.X = np.ndarray((nbins**len(centers),len(centers)))
        self.y = np.ndarray((nbins**len(centers)))
        self.y = counts.flatten()
        
        temp = np.meshgrid(*centers)
        for i in range(len(centers)):
            self.X[:,i] = temp[i].flatten()
        
        


    def train(self):
        """
        Train a gaussian process on the preprocessed histogram in order to determine P(keyatt|predatt)

        """

        #Poisson fractional errors on histograms, hopefully won't screw with GP assumptions
        nug = 1/np.sqrt(self.y)
        nug[(nug!=nug) | (nug==np.inf)] = 0.0
        ii = np.where(nug!=0.0)
        gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, nugget=nug[ii])

        try:
            gp.fit(self.X[ii], self.y[ii])
        except:
            print('*****Fit failed*****')
            pass
        
        self.gp = gp
        


    def predict(self, fvec):
        
        pass

    def vismodel(self):
        ii = np.where(self.y!=0)
        f, (ax1, ax2) = plt.subplots(2)
        pred = self.gp.predict(self.X[ii])
        sns.interactplot(self.X[ii][:,0], self.X[ii][:,1], pred, ax=ax1, scatter_kws={"alpha":0.0}, filled=True)
        sns.interactplot(self.X[ii][:,0], self.X[ii][:,1], self.y[ii], ax=ax2, scatter_kws={"alpha":0.0}, filled=True)
        
        





    


    
