from __future__ import print_function, division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from bisect import bisect_left
from sklearn import gaussian_process, ensemble, tree, mixture
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, make_scorer
from abc import ABCMeta, abstractmethod
from astroML.density_estimation import bayesian_blocks
import random
import munge
import pickle
try:
    import triangle
    hasTriangle = True
except ImportError, e:
    hasTriangle = False
    pass


class Model:
    __metaclass__ = ABCMeta

    def __init__(self, hfeatures, pfeatures, pred, pmod=None, store=False, lstep=None, **kwargs):

        #If no pickle file provided, use features to construct object
        if pmod==None:
            self.hfeatures = hfeatures
            self.pfeatures = pfeatures
            self.pred = pred
            self.pred_dtype = pred.dtype
            self.feat_dtype = hfeatures.dtype
            
            if len(hfeatures.shape) == 2:
                self.nfeat = hfeatures.shape[1]
            else:
                self.nfeat = 1
            if len(pred.shape) == 2:
                self.npred = pred.shape[1]
            else:
                self.npred = 1

            self.store = store
            self.lstep = lstep
            for kwarg in kwargs.keys():
                setattr(self, kwarg, kwargs[kwarg])
                
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

    def clean_pred(self, key='M200b'):
        """
        Get rid of zero mass and very low mass entries in key
        """
        if key=='M200b':
            ii, = np.where((self.pred[key]!=0) & (self.pred[key]>1e10))
        else:
            ii, = np.where(self.pred[key]!=0)
        self.pred = self.pred[ii]
        self.hfeatures = self.hfeatures[ii]

    def feature_dist(self):
        
        pcounts, e = np.histogramdd(self.pfeatures.view((np.float64, len(self.pfeatures.dtype.names))), bins=self.edges[:-1])
        hcounts, e = np.histogramdd(self.hfeatures.view((np.float64, len(self.hfeatures.dtype.names))), bins=self.edges[:-1])

        #Probability that a halo is present at a particle with features in a particular bin of feature space
        #is the number of halos in that bin of feature space over the number of particles in that bin of 
        #feature space

        self.php = hcounts/pcounts

    def assignHalo(self, fvec):
        ii = np.ndarray(fvec.shape[1], dtype=int)
        for j in range(len(ii)):
            bi = bisect_left(self.edges[j], fvec[j])
            #If higher than largest bin edge, set as last bin
            if bi>(len(self.edges[j])-2):
                bi = len(self.edges[j])-2
                ii[j] = bi

        #If random draw less than probability of assigning
        #halo to particle with this feature vector then
        #assign halo                                                                        
        draw = random.random()
        if draw<=self.php[ii]:
            return True
        else:
            return False

    def log_binning(self, histarray, dex):
        """
        Create log spaced bins for the array in histarray with
        size dex. 

        Inputs:
        histarray, np.ndarray -- two dimensional array, whose
        columns we wish to bin
        
        dex, np.ndarray or float -- one dimensional array or 
        float with the size of the bins desired for each column
        with length equal to the number of columns in histarray
        or a float if same spacing for all cols

        Outputs:
        bins, list -- list of arrays, each one containing 
        the bin edges for a column
        """

        cmin = np.min(histarray, axis=0)
        cmax = np.max(histarray, axis=0)
        nbins = np.ceil((np.log10(cmax)-np.log10(cmin))/dex)
        print(nbins)
        bins = [np.logspace(np.log10(cmin[i]), np.log10(cmax[i]), nbins[i]) for i in range(len(cmin))]

        return bins

    def histogram(self, histarray, normed=True, binning='log'):
        """
        Create a histogram using the provided array
        """

        if binning == 'log':
            if self.lstep == None:
                self.lstep = np.zeros(histarray.shape[1]) + 0.01

            bins = self.log_binning(histarray, self.lstep)
        else:
            print('Binning type not understood, aborting!')
            raise

        counts, edges = np.histogramdd(histarray, bins=bins, normed=normed)

        return counts, bins

    def flattenHist(self, counts, edges):
        """
        Given an array of counts, and the bin edges they correspond to 
        flatten the edges into a m x n-features array and the counts
        into a m x 1 array 
        """
        centers = [[(edges[i][j]+edges[i][j+1])/2 for j in range(len(edges[i])-1)] for i in range(len(edges))]

        npts = 1
        for i in range(len(centers)):
            npts*=len(centers[i])

        X = np.ndarray((npts,len(centers)))
        y = counts.flatten()

        temp = np.meshgrid(*centers)
        for i in range(len(centers)):
            X[:,i] = temp[i].flatten()

        return X, y, edges


    def visDensity(self, X, d, nslices=3, f=None, ax=None, label=None, suptitle=None):

        if nslices!=None:
            sX = np.sort(np.unique(X[:,0]))
            step = np.ceil((sX.shape[0]+nslices-1)/nslices)
            dbins = sX[::step]
            #dbins = np.logspace(np.log10(sX[0]), np.log10(sX[-1]), num=nslices**2)
        else:
            dbins = np.unique(X[:,0])
            nslices = len(dbins)
            
        if (f==None) and (ax==None):
            f, ax = plt.subplots(nslices,nslices)
        
        for i, db in enumerate(dbins):
            if i==(len(dbins)-1): continue
            #xii = np.where((dbins[i]<=X[:,0]) & (X[:,0]<dbins[i+1]))
            xii = np.where(dbins[i]==X[:,0])
            dens = d[xii]
            widths = X[xii][:,1][1:]-X[xii][:,1][:-1]
            ax[i/nslices,i%nslices].set_xscale("log", nonposx='clip')
            ax[i/nslices,i%nslices].bar(X[xii][:,1][:-1],dens[:-1],width=widths,label=label)
            ax[i/nslices,i%nslices].set_title('Density = {0:.2f}'.format(db))

        plt.tight_layout()

        if suptitle!=None:
            plt.subplots_adjust(top=0.85)            
            f.suptitle(suptitle)

        return f, ax

        
class HistGauss(Model):

    def preprocess(self):
        """
        Preprocess the data that we will fit the model to

        """
        self.preproc_hist()
        self.feature_dist()

        if self.store==True:
            self.hfeatures=None
            self.pfeatures=None
            self.pred = None
        
    def preproc_hist(self):
        
        self.clean_pred(key=self.pred.dtype.names[0])
        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        histarray = histarray.view((np.float, len(histarray.dtype.names)))        

        counts, edges = self.histogram(histarray, normed=True)
        self.X, self.y, self.edges = self.flattenHist(counts, edges)

        
    def train(self, cv=None):
        """
        Train a gaussian process on the preprocessed histogram in order to determine P(keyatt|predatt)

        """

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
            reg = gaussian_process.GaussianProcess(theta0=[1e12,1], thetaL=[1e10,1e-2], thetaU=[1e14,1e1], nugget=1e-5)

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

    def train(self, cv=None, n_jobs=1):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(\
            self.X, self.y, test_size=0.5, random_state=0)

        if cv!=None:
            param_grid = {'n_estimators': [5, 10, 20, 40], 'max_features': ['sqrt']}
            scorer = make_scorer(r2_score)
            rndf = ensemble.RandomForestRegressor()
            reg = GridSearchCV(rndf,param_grid,cv=cv,scoring=scorer,n_jobs=n_jobs)
            reg.fit(self.X_train, self.y_train)
        else:
            try:
                reg = ensemble.RandomForestRegressor(n_estimators=20, n_jobs=n_jobs)
                reg.fit(self.X_train,self.y_train)
                
            except Exception as e:
                print(e)
                print('*****Fit Failed*****')
        
        print('Fit successful')
        self.reg = reg

        if self.store==True:
            self.hfeatures = None
            self.pfeatures = None
            self.pred = None
            
    def binedges(self):
        """
        Calculate feature bin edges for feature distribution calculation
        """
        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        histarray = histarray.view((np.float, len(histarray.dtype.names)))
        counts, edges = self.histogram(histarray, normed=True)
        X, y, edges = self.flattenHist(counts, edges)
        self.edges = edges

    def preprocess(self):
        """
        Clean the data
        """
        self.clean_pred()
        self.X = np.atleast_2d(self.hfeatures.view((float, len(self.hfeatures[0])))).T
        self.y = self.pred['M200b']
        self.binedges()
        self.feature_dist()

    def predict(self, fvec):
        """                                                                                                                                                               Use the model to predict values using the provided feature vector
        """
        
        return self.reg.predict(fvec)

    def visModel(self):

        f, ax = plt.subplots(2)
        pred = self.reg.predict(self.X_test)

        histarray = np.vstack((self.X_test.T, pred.T)).T
        counts, edges = self.histogram(histarray)
        X, y, edges = self.flattenHist(counts, edges)
        f, ax = self.visDensity(X, y, label='Pred')

        histarray = np.vstack((self.X_test.T, self.y_test.T)).T
        counts, edges = self.histogram(histarray)
        X, y, edges = self.flattenHist(counts, edges)
        f, ax = self.visDensity(X, y, f=f, ax=ax, label='Truth')

        plt.legend()


class pdfRF(Model):

    def preprocess(self):
        """
        Preprocess the data that we will fit the model to

        """
        self.preproc_hist()
        self.feature_dist()

        if self.store==True:
            self.hfeatures=None
            self.pfeatures=None
            self.pred = None
        
    def preproc_hist(self):
        self.clean_pred()
        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        histarray = histarray.view((np.float, len(histarray.dtype.names)))        

        pdf, self.edges = self.histogram(histarray, normed=True)
        self.support, self.jpdf, self.edges = self.flattenHist(pdf, self.edges)
        self.X = np.atleast_2d(self.hfeatures.view((float, len(self.hfeatures[0])))).T
        self.y = self.make_labels()
        
    def make_labels(self):
        self.classes = (self.edges[-1][:-1]+self.edges[-1][1:])/2
        label = np.digitize(self.pred.view((np.float64, len(self.pred[0]))),\
 self.edges[-1])
        #move values outside of bins into first/last bins
        label[label==len(self.edges[-1])] = len(self.edges[-1])-1
        label -= 1
        label[label<0] = 0

        return label

    def train(self, cv=None, n_jobs=1):
        """
        Fit a predictive model to features and pred
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=0)

        if cv!=None:
            param_grid = {'n_estimators': [5, 10, 20, 40], 'max_features': ['sqrt']}
            scorer = make_scorer(r2_score)
            rndf = ensemble.RandomForestClassifier()
            reg = GridSearchCV(rndf,param_grid,cv=cv,scoring=scorer,n_jobs=n_jobs)
            reg.fit(self.X_train, self.y_train)
        else:
            try:
                reg = ensemble.RandomForestClassifier(n_estimators=20, n_jobs=n_jobs)
                reg.fit(self.X_train,self.y_train)
                
            except Exception as e:
                print(e)
                print('*****Fit Failed*****')
                raise
        
        print('Fit successful')
        self.reg = reg

        if self.store==True:
            self.X = None
            self.y = None
            self.pred = None

    def predict(self, fvec):
        pdf = self.reg.predict_proba(fvec)
        cdf = np.cumsum(pdf, axis=1)
        draw = np.random.random(len(fvec))
        ii = np.array([bisect_left(cdf[i],draw[i]) for i in range(len(fvec))])
        
        return self.classes[ii]

    def visModel(self):

        f, ax = plt.subplots(2)
        pred = self.predict(self.X_test)

        histarray = np.vstack((self.X_test.T, pred.T)).T
        counts, edges = self.histogram(histarray)
        X, y, edges = self.flattenHist(counts, edges)
        f, ax = self.visDensity(X, y, label='Pred')

        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        counts, edges = self.histogram(histarray)
        X, y, edges = self.flattenHist(counts, edges)
        f, ax = self.visDensity(X, y, f=f, ax=ax, label='Truth')

        plt.legend()


class DT(Model):

    def train(self, cv=None, n_jobs=1):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(\
            self.X, self.y, test_size=0.5, random_state=0)

        if cv!=None:
            param_grid = {'n_estimators': [5, 10, 20, 40], 'max_features': ['sqrt']}
            scorer = make_scorer(r2_score)
            rndf = tree.DecisionTreeRegressor()
            reg = GridSearchCV(rndf,param_grid,cv=cv,scoring=scorer,n_jobs=n_jobs)
            reg.fit(self.X_train, self.y_train)
        else:
            try:
                reg = tree.DecisionTreeRegressor()
                reg.fit(self.X_train,self.y_train)
                
            except Exception as e:
                print(e)
                print('*****Fit Failed*****')
        
        print('Fit successful')
        self.reg = reg

        if self.store==True:
            self.hfeatures = None
            self.pfeatures = None
            self.pred = None
            
    def binedges(self):
        """
        Calculate feature bin edges for feature distribution calculation
        """
        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        histarray = histarray.view((np.float, len(histarray.dtype.names)))
        counts, edges = self.histogram(histarray, normed=True)
        X, y, edges = self.flattenHist(counts, edges)
        self.edges = edges

    def preprocess(self):
        """
        Clean the data
        """
        self.clean_pred()
        self.X = np.atleast_2d(self.hfeatures.view((float, len(self.hfeatures[0])))).T
        self.y = self.pred['M200b']
        self.binedges()
        self.feature_dist()

    def predict(self, fvec):
        """                                                                                                                                                               Use the model to predict values using the provided feature vector
        """
        
        return self.reg.predict(fvec)

    def visModel(self):

        f, ax = plt.subplots(2)
        pred = self.reg.predict(self.X_test)

        histarray = np.vstack((self.X_test.T, pred.T)).T
        counts, edges = self.histogram(histarray)
        X, y, edges = self.flattenHist(counts, edges)
        f, ax = self.visDensity(X, y, label='Pred')

        histarray = np.vstack((self.X_test.T, self.y_test.T)).T
        counts, edges = self.histogram(histarray)
        X, y, edges = self.flattenHist(counts, edges)
        f, ax = self.visDensity(X, y, f=f, ax=ax, label='Truth')

        plt.legend()



class pdfDT(Model):

    def preprocess(self):
        """
        Preprocess the data that we will fit the model to

        """
        self.preproc_hist()
        self.feature_dist()

        if self.store==True:
            self.hfeatures=None
            self.pfeatures=None
            self.pred = None
        
    def preproc_hist(self):
        self.clean_pred(key=self.pred.dtype.names[0])
        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        histarray = histarray.view((np.float, len(histarray.dtype.names)))        

        pdf, self.edges = self.histogram(histarray, normed=True)
        self.support, self.jpdf, self.edges = self.flattenHist(pdf, self.edges)
        self.X = np.atleast_2d(self.hfeatures.view((float, len(self.hfeatures[0])))).T
        self.y = self.make_labels()
        
    def make_labels(self):
        self.classes = (self.edges[-1][:-1]+self.edges[-1][1:])/2
        label = np.digitize(self.pred.view((np.float64, len(self.pred[0]))), \
                                self.edges[-1])
        #move values outside of bins into first/last bins
        label[label==len(self.edges[-1])] = len(self.edges[-1])-1
        label -= 1
        label[label<0] = 0

        return label

    def train(self, cv=None, n_jobs=1):
        """
        Fit a predictive model to features and pred
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=0)

        if cv!=None:
            param_grid = {'n_estimators': [5, 10, 20, 40], 'max_features': ['sqrt']}
            scorer = make_scorer(r2_score)
            dt = tree.DecisionTreeClassifier()
            reg = GridSearchCV(dt,param_grid,cv=cv,scoring=scorer,n_jobs=n_jobs)
            reg.fit(self.X_train, self.y_train)
        else:
            try:
                reg = tree.DecisionTreeClassifier()
                reg.fit(self.X_train,self.y_train)
                
            except Exception as e:
                print(e)
                print('*****Fit Failed*****')
                raise
        
        print('Fit successful')
        self.reg = reg

        if self.store==True:
            self.X = None
            self.y = None
            self.pred = None

    def predict(self, fvec):
        pdf = self.reg.predict_proba(fvec)
        cdf = np.cumsum(pdf, axis=1)
        draw = np.random.random(len(fvec))
        ii = np.array([bisect_left(cdf[i],draw[i]) for i in range(len(fvec))])
        
        return self.classes[ii]

    def visModel(self):

        f, ax = plt.subplots(2)
        pred = self.predict(self.X_test)

        histarray = np.vstack((self.X_test.T, pred.T)).T
        counts, edges = self.histogram(histarray)
        X, y, edges = self.flattenHist(counts, edges)
        f, ax = self.visDensity(X, y, label='Pred')

        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        counts, edges = self.histogram(histarray)
        X, y, edges = self.flattenHist(counts, edges)
        f, ax = self.visDensity(X, y, f=f, ax=ax, label='Truth')

        plt.legend()


class GMM(Model):

    def feature_dist(self):
        """
        Fit a GMM to the distribution of particle features
        """
        pX =  self.pfeatures.view((np.float, len(self.pfeatures.dtype.names)))
        pGMM = mixture.GMM(n_components=self.n_components, covariance_type='full')
        pGMM.fit(pX)
        self.pGMM = pGMM
        self.pwCDF = pGMM.weights_.cumsum()
        self.nParticles = len(pX)
        

    def assignHalo(self, fvec):
        """
        Marginalize fitted GMM to get P(halo exists | fvec)
        """
        rand = np.random.random(3)
        pComp = self.pwCDF.searchsorted(rands[0], side='right')
        hComp = self.hwCDF.searchsorted(rands[1], side='right')

        pMVN = sp.stat.multivariate_normal(self.pGMM.means_[pComp,:],
                                           self.pGMM.covars_[pComp,:])
        hMVN = sp.stat.multivariate_normal(self.reg.means_[hComp,:nfeat],
                                           self.featcov)
        
        pDens = pMVN.rvs(size=1)
        hDens = hMVN.rvs(size=1)

        php = hDens*self.nHalos/(pDens*self.nParticles)
        
        if rand[2] <= php:
            return True
        else:
            return False

    def preprocess(self):
        """
        Preprocess the data that we will fit the model to

        """
        if not hasattr(self, 'n_components'):
            raise AttributeError("n_components was not defined! Please define upon model instantiation")
        self.preproc_hist()
        self.feature_dist()

        if self.store==True:
            self.hfeatures=None
            self.pfeatures=None
            self.pred = None
        
    def preproc_hist(self):
        self.clean_pred(key=self.pred.dtype.names[0])
        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        self.X = histarray.view((np.float, len(histarray.dtype.names)))
        self.nHalos = len(self.X)
        pdf, self.edges = self.histogram(self.X, normed=True)

    def train(self):
        self.X_train, self.X_test = train_test_split(self.X, test_size=0.1, random_state=0)
        
        try:
            reg = mixture.GMM(n_components=self.n_components, covariance_type='full')
            reg.fit(self.X_train)
            
        except Exception as e:
            print(e)
            print('*****Fit Failed*****')
            raise
        
        print('Fit successful')
        self.reg = reg
        self.icovars = np.linalg.inv(reg.covars_)
        self.predcov = np.linalg.inv(self.icovars[:, self.nfeat:, self.nfeat:])
        self.featcov = self.reg.covars_[:, :self.nfeat, :self.nfeat]
        self.wCDF = np.cumsum(self.reg.weights_)

    def predict(self, fvec):
        #condition GMM on given features
        lil = np.array([\
                np.dot(self.predcov[i,:,:], self.icovars[i, self.nfeat:, :self.nfeat])\
                    for i in range(self.n_components)])
        mud = np.array([fvec - self.reg.means_[i,:self.nfeat] \
                            for i in range(self.n_components)])
        cmo = np.array([\
                np.dot(lil[i,:,:], mud[i,:,:].T) for i in range(self.n_components)])
        condMeans = np.array([\
                self.reg.means_[i,self.nfeat:] - cmo[i,:,:]
                for i in range(self.n_components)])

        mvn = [sp.stats.multivariate_normal(self.reg.means_[i,:self.nfeat], self.featcov[i,:,:])\
                   for i in range(self.n_components)]
        fsamples = np.array([mvn[i].pdf(fvec) for i in range(self.n_components)])

        if len(fvec)==1:
            fsamples = np.atleast_2d(fsamples).T

        condWeights = np.array([\
                self.reg.weights_[i]*fsamples[i,:]
                for i in range(self.n_components)])
        condWeights = condWeights/np.sum(condWeights, axis=0)

        #Sample from conditional distribution
        #first select the component to associate input with
        X = np.empty((len(fvec), self.npred))
        weightCDF = np.cumsum(condWeights, axis=0)
        rand = np.random.random(len(fvec))
        comps = np.array([weightCDF[:,i].searchsorted(rand[i], side='right')\
                              for i in range(len(fvec))])

        #create conditional distributions to draw from using new means
        #and covs
        mvn = [sp.stats.multivariate_normal(condMeans[c,:,i], self.predcov[c,:,:])\
                    for i, c in enumerate(comps)]
        
        #Sample each component associated with an input feature
        for i in range(len(comps)):
            X[i] = mvn[i].rvs(size=1)
                    
        return X

    def visModel(self, labels=None, fname=None):

        nSamples = 1e6
        samples = self.reg.sample(n_samples=nSamples)
        
        if self.pred == None:
            if hasTriangle:
                figure = triangle.corner(samples, labels=labels,
                                         quantiles=[0.16, 0.5, 0.84],
                                         show_titles=True, title_args={"fontsize": 12})
                if fname!=None:
                    plt.savefig('predicted_'+fname)

            elif samples.size[1]==2:
                f, ax = plt.subplots(1)
                ax.hist2d(samples[:,0], samples[:,1])
                if fname!=None:
                    plt.savefig('predicted_'+fname)

            else:
                raise NotImplementedError("Plotting datasets w/ dim > 2 without Triangle not implemented")

        else:
            if hasTriangle:
                figure = triangle.corner(samples, labels=labels,
                                         quantiles=[0.16, 0.5, 0.84],
                                         show_titles=True, title_args={"fontsize": 12})
                if fname!=None:
                    plt.savefig('predicted_'+fname)
                
                figure = triangle.corner(self.X, labels=labels,
                                         quantiles=[0.16, 0.5, 0.84],
                                         show_titles=True, title_args={"fontsize": 12})
                if fname!=None:
                    plt.savefig('original_'+fname)

            elif samples.size[1]==2:
                f, ax = plt.subplots(2)
                ax[0].hist2d(samples[:,0], samples[:,1])
                ax[1].hist2d(self.X[:,0], self.X[:,1])
                ax[1].set_xlabel(labels[0])
                ax[1].set_ylabel(labels[1])
                plt.tight_layout()
                if fname!=None:
                    plt.savefig(fname)

            else:
                raise NotImplementedError("Plotting datasets w/ dim > 2 without Triangle not implemented")

    def visCPDF(self, condition):

        raise NotImplementedError
        
        
        
        
        
