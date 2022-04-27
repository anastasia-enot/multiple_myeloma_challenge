import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn import svm
import xgboost as xgb

rnaseq = pd.read_csv('data/only_1_BM_all_genes.csv')
rnaseq = rnaseq.drop('num_zeros',axis=1)

new = []
for col in rnaseq.columns:
    new_c = col.replace('_1_BM', '')
    #new_c = new_c.replace('_2_BM', '')
    new.append(new_c)
rnaseq.columns = new

rnaseq = rnaseq.T
print(rnaseq.head())
print('rnaseq shape: ', rnaseq.shape)

print('------------')



add_features = pd.read_csv('data/CLEAN_sc3_Training_ClinAnnotations.csv')
add_features = add_features.drop(add_features[add_features['HR_FLAG'] == 'CENSORED'].index)
add_features = add_features.set_index('Patient')
print('add features shape: ', add_features.shape)
Y = add_features[['HR_FLAG']]

Y['HR_FLAG'] = Y['HR_FLAG'].apply(lambda x: 1 if x == 'TRUE' else 0)
print('Y shape: ', Y.shape)

## The data with features

add_features = add_features.drop('HR_FLAG', axis=1)
print('head of add features: ', add_features.head())
print(len(add_features))
print(len(rnaseq))
X_copy = rnaseq.copy()
# Take expression data only if it is in the patient clinical data
rnaseq = rnaseq[rnaseq.index.isin(add_features.index)]
print('new rnaseq new shape: ', rnaseq.shape)

feat_list = add_features.index.to_list()
rna_list = rnaseq.index.to_list()
Y = Y[Y.index.isin(rnaseq.index)]
print(len(rnaseq))
X = pd.concat([rnaseq, add_features], axis=1, join="inner")
X['D_Gender'] = X['D_Gender'].apply(lambda x: 1 if x == 'Female' else 0)
print(X.head())
print('X shape: ', X.shape)
print('Y shape: ', Y.shape)
print('FEATURE SELECTION')
# ANOVA feature selection for numeric input and categorical output
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# define feature selection
fs = SelectKBest(score_func=f_classif, k=100)
# apply feature selection
X_selected = fs.fit_transform(X, Y)
print(pd.DataFrame(X_selected).head())
print(X_selected.shape)
X_selected = pd.DataFrame(X_selected)
X_selected = X_selected.set_index(X.index)

print('------------')

X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size = 0.2, stratify = Y, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print('-------------------------')
print('-------------------------')

apply_PCA = False
if apply_PCA == True:

    print('AFTER PCA')
    from sklearn.decomposition import PCA
    pca = PCA(random_state=20)
    pca.fit(X_copy)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    # plt.vlines(x=80, ymax=1, ymin=0, colors="r", linestyles="--")
    # plt.hlines(y=0.95, xmax=120, xmin=0, colors="g", linestyles="--")
    plt.plot(explained_variance)
    plt.show()

    # We need 13 PC to explain 95% of variance
    pca = PCA(n_components=500)
    pca_X = pca.fit_transform(X_copy)
    pca_X = pd.DataFrame(pca_X)
    pca_X = pca_X.set_index(X_copy.index)
    pca_X_test = pca_X[pca_X.index.isin(X_test.index)]
    print('pca_X_test', pca_X_test.shape)
    pca_X_train = pca_X[pca_X.index.isin(X_train.index)]
    add_features_test = add_features[add_features.index.isin(X_test.index)]
    add_features_train = add_features[add_features.index.isin(X_train.index)]
    print('add_features_test', add_features_test.shape)
    X_test = pd.concat([pca_X_test, add_features_test],axis=1, join="inner")
    X_train = pd.concat([pca_X_train, add_features_train],axis=1, join="inner")
    print(X_test.shape)
    print(X_train.shape)
    #pca_X = pca.transform(X_test)

def train_model(model, X_train, X_test):
    model.fit(X_train, Y_train)
    Y_test_pred = model.predict(X_test)
    print(f'For the model {model}: ')
    print("Accuracy on test set: {:.3}".format(accuracy_score(Y_test, Y_test_pred)))
    print("Recall-score on test set: {:.3}".format(recall_score(Y_test, Y_test_pred)))
    print("Precision-score on test set: {:.3}".format(precision_score(Y_test, Y_test_pred)))
    print("F1-score on test set: {:.3}".format(f1_score(Y_test, Y_test_pred)))
    print(f'AUC score on test set: {roc_auc_score(Y_test, Y_test_pred)}')
    print('-----------------------')
    print('-----------------------')

models = [LogisticRegression(), xgb.XGBClassifier(use_label_encoder=False, n_jobs=7), RandomForestClassifier(), svm.SVC()]

for model in models:
    train_model(model, X_train, X_test)









