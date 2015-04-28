from __future__ import print_function, division
import numpy as np
import scipy as sp
from bisect import bisect_left
from sklearn import gaussian_process
from sklearn import ensemble
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, make_scorer
from abc import ABCMeta, abstractmethod
import random
from astroML.density_estimation import bayesian_blocks
import seaborn as sns
sns.set_context("talk")
sns.despine(left=True)
import matplotlib.pyplot as plt
from matplotlib import cm
import munge
import pickle 


class Model:
    __metaclass__ = ABCMeta

    def __init__(self, hfeatures, pfeatures, pred, pmod=None):

        if pmod==None:
            self.hfeatures = hfeatures
            self.pfeatures = pfeatures
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

    def feature_dist(self):
        
        pcounts, e = np.histogramdd(self.pfeatures.view((np.float, len(self.pfeatures.dtype.names))), bins=self.edges[:-1])
        hcounts, e = np.histogramdd(self.hfeatures.view((np.float, len(self.hfeatures.dtype.names))), bins=self.edges[:-1])

        #Probability that a halo is present at a particle with features in a particular bin of feature space
        #is the number of halos in that bin of feature space over the number of particles in that bin of 
        #feature space

        self.php = hcounts/pcounts

    def assignHalo(self, fvec):
        
        choice = random.random()
        ii = []
        for i in range(self.php.shape[1]):
            ii.append(bisect_left(edges[i], fvec[i]))
        
        ii = np.array(ii)
        if choice<=self.php[ii]:
            return True
        else:
            return False

        
class HistGauss(Model):


    def preprocess(self):
        """
        Preprocess the data that we will fit the model to

        """
        self.preproc_hist()
        
    
    def preproc_hist(self, normed=True):
        """
        Create a histogram from the feature data and data to predict
        
        """
        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        histarray = histarray.view((np.float, len(histarray.dtype.names)))

        bins = self.adaptive_binning(histarray)
        counts, edges = np.histogramdd(histarray, bins=bins, normed=normed)

        self.edges = bins
        centers = [[(edges[i][j]+edges[i][j+1])/2 for j in range(len(edges[i])-1)] for i in range(len(edges))]

        npts = 1
        for i in range(len(centers)):
            npts*=len(centers[i])

        self.X = np.ndarray((npts,len(centers)))
        self.y = counts.flatten()
        
        temp = np.meshgrid(*centers)
        for i in range(len(centers)):
            self.X[:,i] = temp[i].flatten()
        
        self.hfeatures = None
        self.pred = None

        
    def adaptive_binning(self, histarray):
        """
        Calculate optimal bin edges for density histogram using adaptive bayesian blocks 
        """
        
        bins = []

        for i in range(len(histarray.shape)):
            #subsample the data to speed up adaptive binning
            t = random.sample(histarray[:,i],100000)
            #t = histarray[:,i]

            b = bayesian_blocks(t)
            bins.append(b)

        return bins
        
        
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
            reg = gaussian_process.GaussianProcess(theta0=[1e12,1], thetaL=[1e10,1e-1], thetaU=[1e14,1e1], nugget=1e-5)

        try:
            reg.fit(self.X_train, self.y_train)
        except Exception as e:
            print(e)
            print('*****Fit failed*****')
            self.reg = reg
            return

        self.reg = reg
        self.integrate_gp()
        #self.feature_dist()

    
    def select_train_test(self):
        """
        Sample data 'optimally', to reduce size of training set
        """
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
        pdf = pdf.reshape(np.shape(grid)[1:][::-1])

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
        

    def visDensity(self, suptitle=None):
        ncuts = 3
        f, ax = plt.subplots(ncuts,ncuts)
        nsel = np.ceil(len(self.edges[0])/(ncuts**2))

        dbins = [self.edges[0][i] for i in range(len(self.edges[0])) if (i%nsel)==0]
        print(dbins)
        print(nsel)
        print(len(self.edges[0]))

        for i,d in enumerate(dbins):
            if i==(len(dbins)-1): continue
            xii = np.where((dbins[i]<=self.X[:,0]) & (self.X[:,0]<dbins[i+1]))
            dens = self.y[xii]

            ax[i/ncuts,i%ncuts].set_xscale("log", nonposx='clip')
            ax[i/ncuts,i%ncuts].plot(self.X[xii][:,1],dens)
            ax[i/ncuts,i%ncuts].set_title('Density = {0:.2f}'.format(d))

        plt.tight_layout()

        if suptitle!=None:
            plt.subplots_adjust(top=0.85)            
            f.suptitle(suptitle)


    def visModel(self, suptitle=None):

        ncuts = 3
        f, ax = plt.subplots(ncuts,ncuts)
        sX = np.sort(np.unique(self.X[:,0]))
        dbins = np.logspace(np.log10(np.min(sX)),np.log10(np.max(sX))/2,ncuts**2+1)
        for i,d in enumerate(dbins):
            if i==(len(dbins)-1): continue
            xii = np.where((dbins[i]<=self.X[:,0]) & (self.X[:,0]<dbins[i+1]))
            pred = self.reg.predict(self.X[xii])
            truth = self.y[xii]

            ax[i/ncuts,i%ncuts].set_xscale("log", nonposx='clip')
            ax[i/ncuts,i%ncuts].plot(self.X[xii][:,1],pred,label='Prediction')
            ax[i/ncuts,i%ncuts].plot(self.X[xii][:,1],truth, label='Truth')
            ax[i/ncuts,i%ncuts].set_title('Density = {0:.2f}'.format(d))
            plt.legend()

        if suptitle!=None:
            f.suptitle(suptitle)

        plt.tight_layout()
        
        

class RF(Model):

    def train(self, cv=None):
        """                                                                                                                                                               Fit a predictive model to features and pred                                                                                                                       """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=0)

        if cv!=None:
            param_grid = {'n_estimators': [5, 10, 20, 40], 'max_features': ['sqrt']}
            scorer = make_scorer(r2_score)
            rndf = ensemble.RandomForestRegressor()
            reg = GridSearchCV(rndf,param_grid,cv=cv,scoring=scorer,n_jobs=-1)
            reg.fit(self.X_train, self.y_train)
            print('Fit successful')
        else:
            try:
                reg = ensemble.RandomForestRegressor(n_jobs=-1)
                reg.fit(self.X_train,self.y_train)
            except Exception as e:
                print(e)
                print('*****Fit Failed*****')

        self.reg = reg


    def preprocess(self):
        """                                                                                                                                                               Clean the data                                                                                                                                                    """
        self.X = np.atleast_2d(self.hfeatures['delta']).T
        self.y = self.pred['M200b']


    def predict(self, fvec):
        """                                                                                                                                                               Use the model to predict values using the provided feature vector                                                                                                 """
        
        return self.reg.predict(fvec)


    def visModel(self):

        f, ax = plt.subplots(2)
        pred = self.reg.predict(self.X_test)

        print('Plotting fit')
        nii = np.random.choice(np.arange(len(pred)), 10000, replace=False)
        ii0, = np.where((pred[nii]==pred[nii]) & (pred[nii]!=0))
        ii1, = np.where((self.y_test[nii]==self.y_test[nii]) & (self.y_test[nii]!=0))
        sns.kdeplot(self.X_test[nii][ii0].ravel(),np.log10(pred[nii][ii0]),label='Prediction',ax=ax[0])
        sns.kdeplot(self.X_test[nii][ii1].ravel(),np.log10(self.y_test[nii][ii1]), label='Truth', ax=ax[1])
        plt.legend()



class KDE(Model):
    
    
    def preprocess(self):

        nbins = 100
        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        self.histarray = histarray.view((np.float, len(histarray.dtype.names)))
        
        counts, edges = np.histogramdd(self.histarray, bins=nbins, normed=True)
        
        self.edges = edges
        centers = [[(edges[i][j]+edges[i][j+1])/2 for j in range(len(edges[i])-1)] for i in range(len(edges))]
        
        npts = 1
        for i in range(len(centers)):
            npts*=len(centers[i])
            
        self.X = np.ndarray((npts,len(centers)))
        self.y = counts.flatten()
        
        temp = np.meshgrid(*centers)
        for i in range(len(centers)):
            self.X[:,i] = temp[i].flatten()


    def train(self, cv=None):

        self.hist_train, self.hist_test = train_test_split(self.histarray, test_size=0.5, random_state=0)
        
        if cv==None:
            bandwidth = 1e8
            #rtol = 1e-8
            kde = KernelDensity(bandwidth=bandwidth)
            kde.fit(self.histarray)
            self.reg = kde
            
        else:
            param_grid = {'bandwidth':np.logspace(1,12,10)}
            kde = GridSearchCV(KernelDensity(),param_grid,cv=cv)
            kde.fit(self.hist_train)


    def visModel(self, suptitle=None):

        ncuts = 3
        f, ax = plt.subplots(ncuts,ncuts)
        sX = np.sort(np.unique(self.X[:,0]))
        dbins = np.logspace(np.log10(np.min(sX)),np.log10(np.max(sX)),ncuts**2+1)
        print(dbins)
        for i,d in enumerate(dbins):
            if i==(len(dbins)-1): continue
            xii = np.where((dbins[i]<=self.X[:,0]) & (self.X[:,0]<dbins[i+1]))
            log_pdf = self.reg.score_samples(self.X[xii])
            pred = np.exp(log_pdf)
            truth = self.y[xii]

            ax[i/ncuts,i%ncuts].set_xscale("log", nonposx='clip')
            ax[i/ncuts,i%ncuts].plot(self.X[xii][:,1],pred,label='Prediction')
            ax[i/ncuts,i%ncuts].plot(self.X[xii][:,1],truth, label='Truth')
            ax[i/ncuts,i%ncuts].set_title('Density = {0:.2f}'.format(d))
            plt.legend()

        if suptitle!=None:
            f.suptitle(suptitle)

        plt.tight_layout()

    
    def predict(self):
        pass

    
