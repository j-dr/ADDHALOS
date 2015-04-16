from __future__ import print_function
import numpy as np
import scipy as sp
from bisect import bisect_left
from sklearn import gaussian_process
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, make_scorer
from abc import ABCMeta, abstractmethod
from random import random
import seaborn as sns
sns.set_context("talk")
sns.despine(left=True)
import matplotlib.pyplot as plt
from matplotlib import cm
import munge
import pickle 

class Model:
    __metaclass__ = ABCMeta

    def __init__(self, features, pred, pmod=None):

        if pmod==None:
            self.features = features
            self.pred = pred
        else:
            raise NotImplementedError
            
            

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
        
    
    def preproc_hist(self, ranges=None, nbins=100, normed=True):
        """
        Create a histogram from the feature data and data to predict
        
        """
        arrays = [self.features, self.pred]
        histarray = munge.join_rec_arrays(arrays)

        counts, edges = np.histogramdd(histarray.view((np.float, len(histarray.dtype.names))),\
                                                        range=ranges, bins=nbins, normed=normed)

        self.edges = edges
        centers = [[(edges[i][j]+edges[i][j+1])/2 for j in range(len(edges[i])-1)] for i in range(len(edges))]
        self.X = np.ndarray((nbins**len(centers),len(centers)))
        self.y = np.ndarray((nbins**len(centers)))
        self.y = counts.flatten()
        
        temp = np.meshgrid(*centers)
        for i in range(len(centers)):
            self.X[:,i] = temp[i].flatten()
        
        self.features = None
        self.pred = None
        
        


    def train(self, cv=None):
        """
        Train a gaussian process on the preprocessed histogram in order to determine P(keyatt|predatt)

        """

        #Poisson fractional errors on histograms, hopefully won't screw with GP assumptions
        ii = np.where(self.y!=0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=0)

        print('np.shape(X_train): {0}'.format(np.shape(self.X_train)))
        print('np.shape(y_train): {0}'.format(np.shape(self.y_train)))

        if cv!=None:
            parameters = {'theta0': [1e-8,1e-5,1e-2], 'thetaL': [1e-6,1e-5,1e-4], 'thetaU': [1e-3,1e-2,1], 'nugget': [1e-10,1e-5,1e-2,1]}
            scorer = make_scorer(r2_score)
            gp = gaussian_process.GaussianProcess()
            reg = GridSearchCV(gp,parameters,cv=cv,scoring=scorer)
        else:
            reg = gaussian_process.GaussianProcess(theta0=[5e9,1], thetaL=[1e9,1e-1], thetaU=[1e10,1e1], nugget=1e-5)

        try:
            reg.fit(self.X_train, self.y_train)
        except Exception as e:
            print(e)
            print('*****Fit failed*****')
            self.reg = reg
            return

        self.reg = reg
        self.integrate_gp()

    
    def select_train_test(self):
        #ii = np.where(self.y!=0)
        

        pass
        
    
    def integrate_gp(self):
        """
        Integrate the GP to get CDF of parameter we want to predict

        """
        centers = [(e[1:]+e[:-1])/2 for e in self.edges]
        self.centers = centers
        grid = np.meshgrid(*centers)
        pX = np.ndarray((len(grid[0].ravel()), len(grid)))
        for i in range(len(grid)):
            pX[:,i] = grid[i].ravel()

        pdf = self.reg.predict(pX)
        pdf = pdf.reshape(np.shape(grid)[1:])

        #Assuming uniform binning
        dx = self.edges[-1][1:]-self.edges[-1][:-1]
        weights = pdf*dx
        
        self.cdf = np.cumsum(weights, axis=-1)
        #self.cdf = self.cdf/self.cdf[:,-1]
        self.cdfgrid = grid

        
        
        

    def predict(self, fvec):
        """
        Use uniform random distribution to sample distribution fit by GP
        fvec must be 2 dimensional
        """

        ii = np.zeros((len(fvec),len(self.cdfgrid)-1))

        for i in range(len(self.cdfgrid)-1):
            s = np.index_exp[:]
            slc = [s[0] if j!=i else 0 for j in range(len(self.cdfgrid))]
            ii[i] = bisect_left(self.cdfgrid[slc],fvec[:,i])
        
        icdf = self.cdf[ii,:]
        rand = [random.random() for i in range(len(fvec))]
        mii = [bisect_left(icdf[i,:],rand[i]) for i in range(len(fvec))]

        return self.centers[-1][mii]
        


    def vismodel(self):

        ncuts = 5
        f, ax = plt.subplots(ncuts,ncuts)
        sX = np.sort(np.unique(self.X[:,0]))

        for i in range(ncuts):
            for j in range(ncuts):
                d = sX[len(sX)*(i*ncuts+j)/(ncuts**2)]
                xii = np.where(self.X[:,0]==d)
            
                pred = self.reg.predict(self.X[xii])
                truth = self.y[xii]
            
                ax[i,j].plot(self.X[xii][:,1],pred,label='Prediction')
                ax[i,j].plot(self.X[xii][:,1],truth, label='Truth')
                ax[i,j].set_title('Density = {0}'.format(d))
                plt.legend()

        plt.tight_layout()
        
        

class RF(Model):

    def train(self, cv=None):
        """                                                                                                                                                               Fit a predictive model to features and pred                                                                                                                       """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=0)

        if cv!=None:
            pass
        else:
            try:
                reg = ensemble.RandomForestRegressor()
                reg.fit(self.X_train,self.y_train)
            except Exception as e:
                print(e)
                print('*****Fit Failed*****')

        self.reg = reg


    def preprocess(self):
        """                                                                                                                                                               Clean the data                                                                                                                                                    """
        self.X = np.atleast_2d(self.features['delta']).T
        self.y = self.pred['M200b']


    def predict(self, fvec):
        """                                                                                                                                                               Use the model to predict values using the provided feature vector                                                                                                 """
        
        return self.reg.predict(fvec)
        

    def vismodel(self):

        f, ax = plt.subplots(2)
        
        pred = self.reg.predict(self.X_test)
        ii = np.random.choice(np.arange(len(pred)), 1000, replace=False)
        sns.kdeplot(self.X_test[ii].ravel(),pred[ii],label='Prediction',ax=ax[0])
        sns.kdeplot(self.X_test[ii].ravel(),self.y_test[ii], label='Truth', ax=ax[1])
        plt.legend()

                    
        
        
        

        




    


    
