from __future__ import print_function, division
from SimulationAnalysis import readHlist as readHL
from fast3tree import fast3tree
import numpy as np
import fitsio

class Hlist:

    def __init__(self, hlistpath, fields=['x', 'y', 'z', 'm200b'], **kwargs):
        
        self.hlistpath = hlistpath
        self.fields = fields
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def loadHalos(self):
        self.halos = readHL(self.hlistpath, fields=self.fields)
        
    def rHalo(self, masscut=5e12, masstag='m200b'):
        """
        Given the positions and masses of a set of halos, calculate the distance
        to the nearest halo above the mass cut specified
        
        Parameters
        ----------
        pos: array_like
            An array of positions in 3D
        mass: array_like
            The masses of the halos
        masscut: float
            The mass above which to find the nearest halo

        """
        sii = self.halos[masstag].argsort()
        mass = self.halos[masstag][sii]
        pos = self.halos[['x', 'y', 'z']][sii]
        del sii
        
        lii = mass.searchsorted(masscut)
        self.rhalo = np.ndarray(len(pos), dtype=np.float64)
        with fast3tree(pos[lii:]) as tree:
            for i, p in enumerate(pos):
                self.rhalo[i] = tree.query_nearest_distance(p)

    def writeFeatures(self, features, outpath, dtype):
        """
        Write the desired features to disk
        Parameters
        ----------
        features: array_like
            Names of the features to write to disk
        outpath: str
            The path to write the features to
        dtype: np.dtype
            The dtype of the output array. Features to write must 
            be the names of the array fields
        """
        out = np.ndarray(len(self.halos), dtype=dtype)

        for i, feature in enumerate(features):
            out[feature] = getattr(self,feature)

        fitsio.write(outpath, out)
