from __future__ import print_function, division
import numpy as np
import pandas as pd
from trainio import *


def histogram(halodata, rngs):
    """
    Create histogram to calculate P(keyattr|predattr,halo present,z)
    where keyattr, predattr can be a tuple of attributes of any length. 

    """
    
    counts = np.histogramdd(




def matchHalos(hlistpath, keyattr, predpath=None):
    """
    Create an array of halos with their associated densities
    
    """


    halos = readHlist(hlistpath)
    hrnn = readHaloRnn(hrnnpath)
    




