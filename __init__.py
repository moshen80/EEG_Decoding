#importing libraries
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# Loading ML Models
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit,cross_validate,train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score, f1_score,make_scorer,classification_report
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

import mne
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
import matplotlib.ticker as ticker
import hdf5storage

from util import *
from models import *



dpath = '../Data_time/Data_Pre.mat'

Xy = hdf5storage.loadmat(dpath)

print(Xy['y'].shape)
print(Xy['X'].shape)


Xt = Xy['X']
yt = Xy['y']
time = Xy['times']
y_df = pd.DataFrame(yt,columns=['A','AV_wN','Pair','V','AV','Noise'])
y_df['index'] = y_df.index


ind = np.where(y_df.Noise.values==0)[0]
V = y_df.V.values
Pair = y_df.Pair.values
X = np.rollaxis(Xt[ind,:,:],2,1)
y = V[ind]

# Electrode clusters 
EEG = np.arange(0,256)
RF = [3, 213, 214, 222, 223]
RC = [162, 163, 172, 180,181]
LC = [65, 70, 71, 75,76]
OC = [127, 116, 117, 126, 138]
EEG_Cluster = RF+RC+LC+OC

cv_fold = 5
cv = StratifiedShuffleSplit(n_splits=cv_fold, test_size=0.4, random_state=0)
svm_clf = svm.SVC(gamma='auto',C=1) # the main model to be test
imp = SimpleImputer(missing_values=np.nan, strategy='median')
clf = make_pipeline(imp,preprocessing.StandardScaler(),svm_clf )
time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='balanced_accuracy', verbose=True)
X_split = np.array_split(X,X.shape[2], axis=2)
            

time_gen = GeneralizingEstimator(clf, n_jobs=-1, scoring='balanced_accuracy',
                                     verbose=True)

accuracy = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=-1)


pickle.dump(accuracy,open('Accuracy_V.p','wb'))
