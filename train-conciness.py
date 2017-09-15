import csv
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import mean_squared_error
seed = 2017
np.random.seed(seed)

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv('data/training/data_train.csv',header=None)
validation = pd.read_csv('data/validation/data_valid.csv',header=None)

#def get_titles(data):
#    return [row[2] for row in data]
#
#vectorizer = CountVectorizer()
#
#training_vectors = vectorizer.fit_transform(get_titles(train.values))

def contains_number(s):
    regex = re.compile("\d")
    if regex.search(s):
        return True
    return False

def cek_out(file):
    if len(file)>=150:
        return True
    return False

def cek_kurang(file):
    if len(file)<=4:
        return True
    return False

def extract_features(filename):
    features = []
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            title = row[2]
            ''' 
                Feel free to create your amazing features here
                ...
            '''
            features.append([len(title),
                             contains_number(title)])
                             #cek_out(title)])
    return np.asarray(features)


# Data loading
X = extract_features("data/training/data_train.csv")
y = np.loadtxt("data/training/conciseness_train.labels", dtype=int)
tagged = pd.read_csv()

#%%
#X_glove = pd.read_csv('data/train_dataset_preprocessed.csv',header=None)
#X_glove.drop([0,1,2,3,4,5,6,7],inplace=True,axis=1)
#    # Model training
#
#X_new = pd.concat([pd.DataFrame(X),X_glove],axis=1)
#
#from imblearn.under_sampling import RandomUnderSampler
#rsm = RandomUnderSampler(random_state=seed)
#X_res,y_res = rsm.fit_sample(X_new,y)


#%%
import xgboost as xgb
xgbo = xgb.XGBClassifier(max_depth=5,n_estimators=200)
rf = RandomForestClassifier(n_estimators=100,random_state=seed)
logit = LogisticRegression()
gb = GradientBoostingClassifier(learning_rate=0.1,n_estimators=500,random_state=2017)
model = logit
from sklearn.model_selection import cross_val_score
result = cross_val_score(model,X,y,cv=5,scoring='roc_auc')
#model.fit(X, y)
#print("Model RMSE: %f" % mean_squared_error(model.predict_proba(X_res)[:,1], y_res)**0.5)
#%%

    # Validation predicting
X_valid = extract_features("data/validation/data_valid.csv")
#X_valid_glove = pd.read_csv('data/phase_one_validation_dataset_preprocessed.csv',
#                            header=None)
#X_valid_glove.drop([0,1,2,3,4,5,6,7],inplace=True,axis=1)
#
#X_val = pd.concat([pd.DataFrame(X_valid),X_valid_glove],axis=1)
predicted_results = model.predict_proba(X_valid)[:, 1]
pred = pd.DataFrame(predicted_results)
pred.to_csv('conciseness_valid.predict',sep='\n',index=False,header=False)
#np.savetxt(r'conciseness_valid.predict',pred.values,fmt='%5f',
#           delimiter='\n',
#           newline='\n')