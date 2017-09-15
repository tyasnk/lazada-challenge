# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:20:09 2017

@author: tyasnuurk
"""

import pandas as pd
import numpy as np
seed = 2017
np.random.seed(seed)
glove_train = pd.read_csv('data/train_dataset_preprocessed.csv',
                          header=None,usecols=[8,9])

glove_val = pd.read_csv('data/phase_one_validation_dataset_preprocessed.csv',
                          header=None,usecols=[8,9])
df = pd.read_csv('train_prepro.csv',header=None)

df_val = pd.read_csv('val_prepro.csv',header=None)

y_train = np.loadtxt("data/training/conciseness_train.labels", dtype=int)

merged = pd.concat([df,df_val])
merged_data = pd.get_dummies(merged)

X_train = merged_data[0:len(df)]
X_val = merged_data[len(df):]

X_train = pd.concat([X_train,glove_train],axis=1)
X_val = pd.concat([X_val,glove_val],axis=1)
