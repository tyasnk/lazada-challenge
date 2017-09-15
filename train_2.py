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
merged.drop([0,2,5,6],inplace=True,axis=1)
merged_data = pd.get_dummies(merged)

X_train = merged_data[0:len(df)]
X_val = merged_data[len(df):]

X_train = pd.concat([X_train,glove_train],axis=1)
X_val = pd.concat([X_val,glove_val],axis=1)
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import xgboost as xgb
logit = LogisticRegression()
svc = SVC(kernel='poly',probability=True)

isf = IsolationForest(n_estimators=200,random_state=seed)
rf = RandomForestClassifier(n_estimators=200,random_state=seed,
                            n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=100,learning_rate=1.,
                                random_state=seed)
xgbo = xgb.XGBClassifier(n_estimators=200,learning_rate=0.1)
model = isf
score = np.mean(cross_val_score(model, X_train,y_train,cv=5,
                        scoring='accuracy'))
#%%
model.fit(X_train,y_train)

pred = model.predict_proba(X_val)[:, 1]
pred = pd.DataFrame(pred)
pred.to_csv('conciseness_valid.predict',sep='\n',index=False,header=False)
