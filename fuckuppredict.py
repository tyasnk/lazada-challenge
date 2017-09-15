# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:44:53 2017

@author: tyasnuurk
"""

import pandas as pd
import numpy as np

clarity = pd.read_csv('clarity_valid.predict',delimiter='\n',header=None)
conciseness = pd.read_csv('conciseness_valid.predict',delimiter='\n',header=None)

def threshhold(X,upper=1,lower=0):
    if X >= upper:
        return 1
    elif X <= lower:
        return 0
    else:
        return X

clarity_new = clarity[0].apply(threshhold,args=(0.95,0.15))
conciseness_new = conciseness[0].apply(threshhold,args=(0.95,0.15)) 

np.savetxt(r'pred/clarity_valid.predict',clarity_new,fmt='%5f',
           delimiter='\n',
           newline='\n')

np.savetxt(r'pred/conciseness_valid.predict',conciseness_new,fmt='%5f',
           delimiter='\n',
           newline='\n')