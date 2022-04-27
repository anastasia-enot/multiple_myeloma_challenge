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
# ANOVA feature selection for numeric input and categorical output
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier

rnaseq = pd.read_csv('data/BM_last_visit_all_genes.csv')
#rnaseq = rnaseq.drop('num_zeros',axis=1)

new_cols = []
for col in rnaseq.columns:
    pat = '_'.join(col.split('_', 2)[0:2])
    new_cols.append(pat)
rnaseq.columns = new_cols

rnaseq = rnaseq.T

add_features = pd.read_csv('data/CLEAN_sc3_Training_ClinAnnotations.csv', sep = ';')
add_features = add_features.drop(add_features[add_features['HR_FLAG'] == 'CENSORED'].index)
add_features = add_features.set_index('Patient')
print('add features shape: ', add_features.shape)
Y = add_features[['HR_FLAG']]

Y['HR_FLAG'] = Y['HR_FLAG'].apply(lambda x: 1 if x == 'TRUE' else 0)
print('Y shape: ', Y.shape)

## The data with features

add_features = add_features.drop('HR_FLAG', axis=1)
add_features['D_Gender'] = add_features['D_Gender'].apply(lambda x: 1 if x == 'Female' else 0)
print('head of add features: ', add_features.head())
print(len(add_features))
print(len(rnaseq))
X_copy = rnaseq.copy()
# Take expression data only if it is in the patient clinical data
rnaseq = rnaseq[rnaseq.index.isin(add_features.index)]
add_features = add_features[add_features.index.isin(rnaseq.index)]
new_cols = ['gene_' + str(col) for col in rnaseq.columns.to_list()]
rnaseq.columns = new_cols
print('new rnaseq new shape: ', rnaseq.shape)
print(rnaseq.head())

# feat_list = add_features.index.to_list()
# rna_list = rnaseq.index.to_list()
Y = Y[Y.index.isin(rnaseq.index)]
Y = Y.values.ravel()
print(len(rnaseq))
X = pd.concat([rnaseq, add_features], axis=1, join="inner")
X = X.values

print('rnaseq shape:', rnaseq.shape)
print('add features shape: ', add_features.shape)
print('X shape: ', X.shape)
print('Y shape: ', Y.shape)

###############################################################
print('FEATURE SELECTION')

# Select k best features from the RNA-seq data
fs = SelectKBest(score_func=f_classif, k=75)
# apply feature selection
rnaseq_selected = fs.fit_transform(rnaseq, Y)
rnaseq_selected = pd.DataFrame(rnaseq_selected)
rnaseq_selected = rnaseq_selected.set_index(rnaseq.index)
cols_1 = ['selected_best_rnaseq_' + str(col) for col in rnaseq_selected.columns.to_list()]
rnaseq_selected.columns = cols_1
print(pd.DataFrame(rnaseq_selected).head())
print(rnaseq_selected.shape)

# Select best features in the clinical annotation dataset. If want to have them all, then set k = 'all'
fs = SelectKBest(score_func=f_classif, k=10)
print('add features shape: ', add_features.shape)
print(add_features.head())
add_features_selected = fs.fit_transform(add_features, Y)

add_features_selected = pd.DataFrame(add_features_selected)
add_features_selected = add_features_selected.set_index(add_features.index)
cols_2 = ['selected_best_clinical_' + str(col) for col in add_features_selected.columns.to_list()]
add_features_selected.columns = cols_2
print(fs.scores_)
print(pd.DataFrame(add_features_selected).head())
print(add_features_selected.shape)

X_selected = pd.concat([rnaseq_selected, add_features_selected], axis=1, join='inner')
X_selected = X_selected.values


print('------------')

X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size = 0.25, stratify = Y, random_state=42)
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
    print(str(model))
    if 'Logistic' in str(model) or 'SVC' in str(model):
        sc_x = StandardScaler()
        X_train = sc_x.fit_transform(X_train)
        X_test = sc_x.transform(X_test)
        print('Standardized!')
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

models = [LogisticRegression(), xgb.XGBClassifier(use_label_encoder=False, n_jobs=7), RandomForestClassifier(),
          svm.SVC(), GaussianNB(), CatBoostClassifier()]

for model in models:
    train_model(model, X_train, X_test)

#
#
#
#
#
#
#
#
