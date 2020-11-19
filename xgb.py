# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 22:45:47 2020

@author: krish
"""


import pandas as pd
import pickle
import tensorflow as tf
print('Using TensorFlow version', tf.__version__)


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import ADASYN
from collections import Counter




credit_data=pd.read_csv("C:/Users/krish/OneDrive/Documents/UoH/datasets tableau project/creditcard.csv")

credit_data_new=credit_data.drop(['Time'],axis=1)

X = credit_data_new.iloc[:, :-1]
y = credit_data_new['Class']
X_train,X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42,stratify=y)
print('train dataset shape {}'.format(Counter(y_train)))
print('test dataset shape {}'.format(Counter(y_test)))


sc=StandardScaler().fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)


# apply the ADASYN over-sampling to handle imbalanced dataset
ada = ADASYN(random_state=42)
print('Original dataset shape {}'.format(Counter(y_train)))
X_res, y_res = ada.fit_resample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))


X1, y1 = X_res, y_res 

X_train1,X_test1,y_train1, y_test1 = train_test_split(X1,y1, test_size=0.2, random_state=42)
X_train1.shape
X_test1.shape
y_train1.shape
y_test1.shape
print('train dataset shape {}'.format(Counter(y_train1)))
print('test dataset shape {}'.format(Counter(y_test1)))

sc=StandardScaler().fit(X_train1)
X_train1=sc.transform(X_train1)
X_test1=sc.transform(X_test1)

from xgboost import XGBClassifier

classifier = XGBClassifier(learning_rate=0.03,n_estimators=2400,num_classes=2,n_jobs=2,objective='binary:logistic',model_class_name='XGBoostGBMModel',random_state=1234,ensemble_level=3,seed=1234,nfolds=5,time_tolerance=2,score_f_name='f1',eval_metric='auc',booster='gbtree')

classifier.fit(X_train1, y_train1, eval_set=[(X_train1, y_train1), (X_test1, y_test1)],early_stopping_rounds=200,verbose=50)


pickle.dump(classifier,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
daf=pd.DataFrame([[-1.359807134,-0.072781173,2.536346738,1.378155224,-0.33832077,0.462387778,0.239598554,0.098697901,
                      0.36378697,0.090794172,-0.551599533,-0.617800856,-0.991389847,-0.311169354,1.468176972,-0.470400525,
                      0.207971242,0.02579058,0.40399296,0.251412098,-0.018306778,0.277837576,-0.11047391,0.066928075,
                      0.128539358,-0.189114844,0.133558377,-0.021053053,149.62]]
)
type(daf)
daf.shape
daf.columns=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']
daf.columns

print(model.predict(daf)[0])
