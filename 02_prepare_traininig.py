import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

rnaseq = pd.read_csv('data/only_1_BM_selected_genes.csv')
rnaseq = rnaseq.drop('num_zeros',axis=1)

new = []
for col in rnaseq.columns:
    new_c = col.replace('_1_BM', '')
    new_c = new_c.replace('_2_BM', '')
    new.append(new_c)
rnaseq.columns = new
col_list = rnaseq.columns.to_list()
print(len(col_list))
col_list = list(set(col_list))
print(len(col_list))
rnaseq = rnaseq.T
print(rnaseq.head())

print('------------')

add_features = pd.read_csv('data/CLEAN_sc3_Training_ClinAnnotations.csv')
add_features = add_features.set_index('Patient')
add_features = add_features.drop('HR_FLAG', axis=1)
X_copy = rnaseq.copy()
rnaseq = pd.concat([rnaseq, add_features], axis=1, join="inner")
rnaseq['D_Gender'] = rnaseq['D_Gender'].apply(lambda x: 1 if x == 'Female' else 0)
print(rnaseq.head())

Y = pd.read_csv('data/sc3_Training_ClinAnnotations.csv')
Y = Y.set_index('Patient')
print(Y.head())

rnaseq['HR_FLAG'] = Y['HR_FLAG']
print(len(rnaseq))

rnaseq = rnaseq.drop(rnaseq[rnaseq['HR_FLAG'] == 'CENSORED'].index)
print(rnaseq['HR_FLAG'].isnull().sum())
rnaseq = rnaseq.dropna()
print(rnaseq.head())
rnaseq['HR_FLAG'] = rnaseq['HR_FLAG'].apply(lambda x: 1 if x == 'TRUE' else 0)
print(rnaseq.head())

print('------------')
# For some reason some patients from RNAseq are not in the file with annotations
print(rnaseq['HR_FLAG'].value_counts())

X = rnaseq.drop(['HR_FLAG'],axis=1)

Y = rnaseq.loc[:, 'HR_FLAG']

print('----------')


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn import svm
import xgboost as xgb
import seaborn as sns

# print('NO PCA')
# print('Log Reg')
# clf = LogisticRegression()
# clf.fit(X_train , Y_train)
# Y_test_pred = clf.predict(X_test)
#
#
# print("Accuracy on test set: {:.3}".format(accuracy_score(Y_test , Y_test_pred)))
# print("Recall-score on test set: {:.3}".format(recall_score(Y_test , Y_test_pred)))
#
# print("Precision-score on test set: {:.3}".format(precision_score(Y_test , Y_test_pred)))
# print("F1-score on test set: {:.3}".format(f1_score(Y_test , Y_test_pred)))
# print(f'AUC score on test set: {roc_auc_score(Y_test, Y_test_pred)}')
# print('-----------------------')
# print('Xgb')
# clf = xgb.XGBClassifier(use_label_encoder=False, n_jobs=7)
# clf.fit(X_train , Y_train)
# Y_test_pred = clf.predict(X_test)
#
# print("Accuracy on test set: {:.3}".format(accuracy_score(Y_test , Y_test_pred)))
# print("Recall-score on test set: {:.3}".format(recall_score(Y_test , Y_test_pred)))
#
# print("Precision-score on test set: {:.3}".format(precision_score(Y_test , Y_test_pred)))
# print("F1-score on test set: {:.3}".format(f1_score(Y_test , Y_test_pred)))
# print(f'AUC score on test set: {roc_auc_score(Y_test, Y_test_pred)}')
#
# print('-----------------------')
# print('SVM')
# clf = svm.SVC()
# clf.fit(X_train , Y_train)
# Y_test_pred = clf.predict(X_test)
#
# print("Accuracy on test set: {:.3}".format(accuracy_score(Y_test , Y_test_pred)))
# print("Recall-score on test set: {:.3}".format(recall_score(Y_test , Y_test_pred)))
#
# print("Precision-score on test set: {:.3}".format(precision_score(Y_test , Y_test_pred)))
# print("F1-score on test set: {:.3}".format(f1_score(Y_test , Y_test_pred)))
# print(f'AUC score on test set: {roc_auc_score(Y_test, Y_test_pred)}')
#
# print('-----------------------')
# print('Random Forest')
# clf = RandomForestClassifier()
# clf.fit(X_train , Y_train)
# Y_test_pred = clf.predict(X_test)
#
# print("Accuracy on test set: {:.3}".format(accuracy_score(Y_test , Y_test_pred)))
# print("Recall-score on test set: {:.3}".format(recall_score(Y_test , Y_test_pred)))
#
# print("Precision-score on test set: {:.3}".format(precision_score(Y_test , Y_test_pred)))
# print("F1-score on test set: {:.3}".format(f1_score(Y_test , Y_test_pred)))
# print(f'AUC score on test set: {roc_auc_score(Y_test, Y_test_pred)}')
#
# cm = metrics.confusion_matrix(Y_test, Y_test_pred)
# sns.heatmap(cm , annot=True , fmt='d')
# plt.show()



