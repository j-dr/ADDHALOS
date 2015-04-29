from __future__ import print_function
import numpy as np


class Config:
    
    def  __init__(self, pdict):

        for key in pdict.keys():
            setattr(self,key,pdict[key])


def readConfigFile(fname):
    
    with open(fname, 'r') as fp:
        
        l = fp.readline()
        keys = []
        values = []

        while l!='':
            l = l.strip()

            if (len(l)==0) or (l[0]=='#'): 
                l = fp.readline()
                continue

            ls = l.split()

            if len(ls)==1:
                print(ls)
                print("Only one columns detected in a row in the config file.\n"\
                      "Please check your formatting")
                raise
            elif len(ls)>2: 
                print("More than two columns in the config file.\n"\
                      "Please check your formatting")
                raise
            
            keys.append(ls[0])
            values.append(ls[1])
            l = fp.readline()

    pdict = dict(zip(keys, values))
    return Config(pdict)

    
