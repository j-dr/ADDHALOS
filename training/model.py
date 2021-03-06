from __future__ import print_function, division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import cm
from bisect import bisect_left
from sklearn import gaussian_process, ensemble, tree, mixture
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, make_scorer
from sklearn.externals import joblib
from abc import ABCMeta, abstractmethod
import random
import munge
import fitsio
import pickle

try:
    import triangle
    hasTriangle = True
except:
    try:
        import corner as triangle
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

            self.nfeat = len(hfeatures.dtype)
            self.npred = len(pred.dtype)

            self.store = store
            self.lstep = lstep
            for kwarg in kwargs.keys():
                setattr(self, kwarg, kwargs[kwarg])
                #if n_pcomponents not given, set to n_components
                if (kwarg=='n_components') and ('n_pcomponents' not in kwargs.keys()):
                    setattr(self, 'n_pcomponents', kwargs[kwarg])

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
        self.pred[key] = np.log10(self.pred[key])
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
            if hasattr(self, 'path'):
                joblib.dump(self, self.path)



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
            if hasattr(self, 'path'):
                joblib.dump(self, self.path)


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
            if hasattr(self, 'path'):
                joblib.dump(self, self.path)


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
            if hasattr(self, 'path'):
                joblib.dump(self, self.path)


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
        if hasattr(self, 'n_pcomponents'):
            n_pcomponents = self.n_pcomponents
        else:
            n_pcomponents = self.n_components
        pGMM = mixture.GMM(n_components=n_pcomponents, covariance_type='full')
        pGMM.fit(pX)
        self.pGMM = pGMM
        self.pwCDF = pGMM.weights_.cumsum()
        self.nParticles = len(pX)


    def assignHalo(self, fvec):
        """
        Marginalize fitted GMM to get P(halo exists | fvec)
        """

        pDens = np.array([sp.stats.multivariate_normal.pdf(
                fvec, mean=self.pGMM.means_[i,:],
                cov=self.pGMM.covars_[i,:])*self.pGMM.weights_[i]
                for i in range(self.n_pcomponents)])
        hDens = np.array([sp.stats.multivariate_normal.pdf(
                fvec, mean=self.reg.means_[i,:self.nfeat],
                cov=self.featcov[i,:,:])*self.reg.weights_[i]
                for i in range(self.n_components)])

        pDens = pDens.sum()
        hDens = hDens.sum()

        php = hDens*self.nHalos/(pDens*self.nParticles)

        if np.random.random() <= php:
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
        if hasattr(self, 'path'):
            joblib.dump(self, self.path)

        if hasattr(self, 'tsetout'):
            self.characterize_tset()

        if self.store==True:
            self.hfeatures=None
            self.pfeatures=None
            self.pred = None



    def characterize_tset(self, labels=None):
        if hasTriangle==False: return
        figure = triangle.corner(self.X, labels=labels,
                                 quantiles=[0.16, 0.5, 0.84],
                                 show_titles=True, title_args={"fontsize": 12})
        plt.savefig(self.tsetout)

    def preproc_hist(self):
        self.clean_pred(key=self.pred.dtype.names[0])
        arrays = [self.hfeatures, self.pred]
        histarray = munge.join_rec_arrays(arrays)
        self.X = histarray.view((np.float, len(histarray.dtype.names)))
        self.nHalos = len(self.X)
        pdf, self.edges = self.histogram(self.X, normed=True)

    def train(self, cv=None):
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
        self.hwCDF = np.cumsum(self.reg.weights_)
        if hasattr(self, 'path'):
            joblib.dump(self, self.path)

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
        #quick fix for better plotting
        samples = np.log10(samples)

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


class MDN(Model):

    def __init__(self, hfeatures, pfeatures, pred, ppred, pmod=None,
                  store=False, lstep=None, n_halo_assignment_epoch=None,
                  halo_assignment_model_path=None, **kwargs):
        super(MDN,self).__init__(hfeatures, pfeatures, pred, pmod=pmod, store=store, lstep=lstep, **kwargs)

        self.ppred = ppred

        self.n_components = kwargs['n_components']
        self.nhidden = int(self.nhidden)
        self.nout = self.n_components*(1 + 2*self.npred)

        if 'batchsize' not in kwargs.keys():
            self.batchsize = 10000
        else:
            self.batchsize = int(kwargs['batchsize'])

        if 'nepoch' not in kwargs.keys():
            self.nepoch = 1
        else:
            self.nepoch = int(kwargs['nepoch'])

        if n_halo_assignment_epoch is None:
            self.n_halo_assignment_epoch = 1
        else:
            self.n_halo_assignment_epoch = int(n_halo_assignment_epoch)

        if halo_assignment_model_path is None:
            self.halo_assignment_model_path = 'MDN_halo_assignment.model'
        else:
            self.halo_assignment_model_path = halo_assignment_model_path

    def trainHaloAssignment(self):
        """
        Fit a MDN to the distribution of particle features
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.pX, self.py, test_size=0.1, random_state=0)

        # Create the model
        x = tf.placeholder(tf.float32, [None, self.nfeats])
        W = tf.Variable(tf.random_normal([self.nfeat, 2], stddev=0.5,
                                         dtype=tf.float32))
        b = tf.Variable(tf.random_normal([2], stddev=0.5, dtype=tf.float32))
        y = tf.matmul(x, W) + b

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 2])

        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        with tf.Session() as sess:
            print('reload')
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()

            #determine number of batches per epoch
            ne = self.X_train.shape[0]
            nb = (ne + self.batchsize - 1)//self.batchsize

            loss = np.zeros(nb*self.nepoch)
            for i in xrange(self.nepoch):
                for j in xrange(nb):
                    fd = {x:self.X_train[j*self.batchsize:(j+1)*self.batchsize],
                          y_:self.y_train[j*self.batchsize:(j+1)*self.batchsize]}
                    sess.run(train_step, feed_dict=fd)
                    loss[i*nb+j] = sess.run(cross_entropy, feed_dict=fd)
                    if (j+1)%100==0:
                        print(loss[i*nb+j])

                print("Loss after {0} epochs:    {1}".format(i+1, loss[i*nb+j]))

                if self.store==True:
                    #save the session after every epoch
                    if hasattr(self, 'halo_assignment_model_path'):
                        save_path = saver.save(sess,
                                                self.halo_assignment_model_path)
                        print("Model saved in file: {0}".format(save_path))


            fd = {x:self.X_test,y_:self.y_test}
            test_loss = sess.run(cross_entropy, feed_dict=fd)

            print('xentropy on test set after full training: {}'.format(test_loss))

        return loss

    def assignHalo(self, fvec):
        """
        Use NN to predict presence of halo
        """

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, self.halo_assignment_model_path)


    def preprocess(self):
        """
        Preprocess the data that we will fit the model to
        """
        if not hasattr(self, 'n_components'):
            raise AttributeError("n_components was not defined! Please define upon model instantiation")

        self.clean_pred(key=self.pred.dtype.names[0])
        self.pX = self.pfeatures.view((np.float, len(self.hfeatures.dtype.names)))
        self.X = self.hfeatures.view((np.float, len(self.hfeatures.dtype.names)))

        self.y = self.pred.view((np.float, len(self.hfeatures.dtype.names)))
        self.py = np.zeros((len(self.ppred),2), dtype=np.int)

        self.py[self.ppred,0]  = 1
        self.py[~self.ppred,1] = 1

        self.pX = np.atleast_2d(self.pX).T

        self.X = np.atleast_2d(self.X).T
        self.y = np.atleast_2d(self.y).T

    def characterize_tset(self, labels=None):
        if hasTriangle==False: return
        figure = triangle.corner(self.X, labels=labels,
                                 quantiles=[0.16, 0.5, 0.84],
                                 show_titles=True, title_args={"fontsize": 12})
        plt.savefig(self.tsetout)

    def train(self, cv=None):

        #split into test and train sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=0)

        x = tf.placeholder("float",shape=[None,self.nfeat])
        y = tf.placeholder("float",shape=[None,self.npred])

        #inputs to hidden layer
        Wxh = tf.Variable(tf.random_normal([self.nfeat, self.nhidden], stddev=0.5, dtype=tf.float32))
        bxh = tf.Variable(tf.random_normal([self.nhidden], stddev=0.5, dtype=tf.float32))

        #hidden layer to outputs
        Who = tf.Variable(tf.random_normal([self.nhidden, self.nout], stddev=0.5, dtype=tf.float32))
        bho = tf.Variable(tf.random_normal([self.nout], stddev=0.5, dtype=tf.float32))

        hh    = tf.nn.tanh(tf.matmul(x,Wxh) + bxh)
        out   = tf.matmul(hh, Who) + bho

        #map nn outputs to gmm parameters
        pi, sigma, mu = self.get_mixture_params(out)

        #minimization step
        lossfcn = self.tf_loglikelihood(pi, sigma, mu, y)
        train_op = tf.train.AdamOptimizer().minimize(lossfcn)

        #start session
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()

            #determine number of batches per epoch
            ne = self.X_train.shape[0]
            nb = (ne + self.batchsize - 1)//self.batchsize

            loss = np.zeros(nb*self.nepoch)
            for i in xrange(self.nepoch):
                for j in xrange(nb):
                    fd = {x:self.X_train[j*self.batchsize:(j+1)*self.batchsize],
                          y:self.y_train[j*self.batchsize:(j+1)*self.batchsize]}
                    sess.run(train_op, feed_dict=fd)
                    if (j+1)%10==0:
                        print(loss[i*nb+j])
                    loss[i*nb+j] = sess.run(lossfcn, feed_dict=fd)

                print("Loss after {0} epochs:    {1}".format(i+1, loss[i*nb+j]))

                if self.store==True:
                    #save the session after every epoch
                    if hasattr(self, 'path'):
                        save_path = saver.save(sess, self.path)
                        print("Model saved in file: {0}".format(save_path))

            tenb = (self.X_test.shape[0] + self.batchsize -1)//self.batchsize
            pred = np.zeros((self.X_test.shape[0], self.nfeat+2*self.npred))
            pred[:,:self.nfeat] = self.X_test
            pred[:,self.nfeat:self.nfeat+self.npred] = self.y_test

            for j in range(tenb):
                out_pi_test, out_sigma_test, out_mu_test = sess.run(self.get_mixture_params(out),
                           feed_dict={x:
                            self.X_test[self.batchsize*j:self.batchsize*(j+1)]})
                pred[self.batchsize*j:self.batchsize*(j+1),
                     self.nfeat+self.npred:] = self.genpred(out_pi_test,
                                                            out_mu_test,
                                                            out_sigma_test)

            fitsio.write(self.path+'.pred', pred)


    def get_mixture_params(self, output):
        pi    = tf.placeholder(dtype=tf.float32, shape=[None,self.n_components])
        sigma = tf.placeholder(dtype=tf.float32, shape=[None,self.npred, self.n_components])
        mu    = tf.placeholder(dtype=tf.float32, shape=[None,self.npred, self.n_components])

        pi_    = tf.slice(output, [0,0], [-1,self.n_components])
        sigma_ = tf.slice(output, [0,self.n_components], [-1,self.npred*self.n_components])
        mu_    = tf.slice(output, [0,self.n_components*(1+self.npred)], [-1,self.npred*self.n_components])

        sigma_ = tf.reshape(sigma_, [-1, self.npred, self.n_components])
        mu     = tf.reshape(mu_, [-1, self.npred, self.n_components])

        max_pi  = tf.reduce_max(pi_, 1, keep_dims=True)
        sub_pi  = tf.exp(tf.sub(pi_, max_pi))
        norm_pi = tf.inv(tf.reduce_sum(sub_pi, 1, keep_dims=True))
        pi      = tf.mul(norm_pi, sub_pi)

        sigma = tf.exp(sigma_)

        return pi, sigma, mu

    def tf_normal(self, y, mu, sigma):
        norm   = 1 / np.sqrt(2*np.pi)
        ytile  = tf.tile(tf.reshape(y,[-1,self.npred,1]),[1,1,self.n_components])
        result = tf.sub(ytile, mu)
        result = tf.mul(result,tf.inv(sigma))
        result = -tf.div(tf.square(result),2)
        result = tf.reduce_sum(result, 1, keep_dims=True)
        detsigma = tf.reduce_sum(sigma, 1, keep_dims=True)
        return tf.mul(tf.mul(tf.exp(result),tf.inv(detsigma)),norm)

    def tf_loglikelihood(self, pi, sigma, mu, y):
        result = self.tf_normal(y, mu, sigma)
        result = tf.mul(result, pi)
        result = tf.reduce_sum(result, 2, keep_dims=True)
        result = -tf.log(result)
        return tf.reduce_mean(result)

    def get_pi_idx(self, x, pdf):
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        return -1

    def genpred(self, out_pi, out_mu, out_sigma):
        NTEST = out_pi.shape[0]
        result = np.random.rand(NTEST, self.npred) # initially random [0, 1]
        rn = np.random.randn(NTEST, self.npred) # normal random matrix (0.0, 1.0)
        mu = np.zeros(self.npred)
        std = np.zeros(self.npred)
        idx = 0

        # transforms result into random ensembles
        for i in range(NTEST):
            idx = self.get_pi_idx(result[i, 0], out_pi[i])
            mu = out_mu[i, :, idx]
            std = out_sigma[i, :, idx]
            result[i, :] = mu + rn[i, :]*std

        return result

    def predict(self, fvec):
        pass

    def visModel(self, labels=None, fname=None):
        pass

    def visCPDF(self, condition):

        raise NotImplementedError
