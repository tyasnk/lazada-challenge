import csv
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
import pandas as pd
seed = 2017
np.random.seed(seed)
#from utils import write_submission

def contains_number(s):
    regex = re.compile("\d")
    if regex.search(s):
        return True
    return False

def extract_features(filename):
    features = []
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            title = row[2]
            desc = row[6]
            ''' 
                Feel free to create your amazing features here
                ...
            '''
            features.append([len(title), len(desc),
                             contains_number(desc),
                             contains_number(title)])
    return np.asarray(features)

X = extract_features("data/training/data_train.csv")
y = np.loadtxt("data/training/clarity_train.labels", dtype=int)

X_glove = pd.read_csv('data/train_dataset_preprocessed.csv',header=None)
X_glove.drop([0,1,2,3,4,5,6,7],inplace=True,axis=1)
    # Model training

X_new = pd.concat([pd.DataFrame(X),X_glove],axis=1)

from imblearn.under_sampling import RandomUnderSampler
rsm = RandomUnderSampler(random_state=seed)
X_res,y_res = rsm.fit_sample(X_new,y)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=seed)
X_bal,y_bal = sm.fit_sample(X_new,y)


import xgboost as xgb
xgbo = xgb.XGBClassifier(max_depth=5,n_estimators=200)
rf = RandomForestClassifier(n_estimators=100,random_state=seed)
logit = LogisticRegression()
gb = GradientBoostingClassifier(learning_rate=0.1,n_estimators=500,random_state=2017)
svc = SVC(kernel='linear',probability=True)


model = rf
model.fit(X_bal, y_bal)
print("Model RMSE: %f" % mean_squared_error(model.predict_proba(X_new)[:,1], y)**0.5)


    # Validation predicting
X_valid = extract_features("data/validation/data_valid.csv")
X_valid_glove = pd.read_csv('data/phase_one_validation_dataset_preprocessed.csv',
                            header=None)
X_valid_glove.drop([0,1,2,3,4,5,6,7],inplace=True,axis=1)

X_val = pd.concat([pd.DataFrame(X_valid),X_valid_glove],axis=1)
predicted_results = model.predict_proba(X_val)[:, 1]
pred = pd.DataFrame(predicted_results)
pred.to_csv('clarity_valid.predict',sep='\n',index=False,header=False)
#np.savetxt(r'clarity_valid.predict',pred.values,fmt='%5f',
#           delimiter='\n',
#           newline='\n')
#write_submission('clarity_valid.predict', predicted_results)