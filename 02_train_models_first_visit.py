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
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier


def prepare_data_for_train(path_expression, path_clinical, num_feat_expression, num_feat_clinical):
    '''
    This function will prepare the data for the training step. The data should already be 'clean': it should contain 
    only Bone Marrow (BM) patients from their 1st visit. (samples ending with 1_BM).
    :param path_expression: (str) the path to expression data. 
    :param path_clinical: (str) the path to clinical data
    :param num_feat_expression: (int) number of
    :return: 
    '''
    
    rnaseq = pd.read_csv('data/BM_first_visit_all_genes.csv')
    #rnaseq = rnaseq.drop('num_zeros',axis=1)
    
    new = []
    for col in rnaseq.columns:
        new_c = col.replace('_1_BM', '')
        new.append(new_c)
    rnaseq.columns = new
    rnaseq = rnaseq.T
    
    clinical_df = pd.read_csv('data/CLEAN_sc3_Training_ClinAnnotations.csv', sep = ';')
    # We do not take into account CENSORED patients
    clinical_df = clinical_df.drop(clinical_df[clinical_df['HR_FLAG'] == 'CENSORED'].index)
    clinical_df = clinical_df.set_index('Patient')
    print('Clinical data shape: ', clinical_df.shape)
    Y = clinical_df[['HR_FLAG']]
    
    Y['HR_FLAG'] = Y['HR_FLAG'].apply(lambda x: 1 if x == 'TRUE' else 0)
    print('Y shape: ', Y.shape)

    
    clinical_df = clinical_df.drop('HR_FLAG', axis=1)
    clinical_df['D_Gender'] = clinical_df['D_Gender'].apply(lambda x: 1 if x == 'Female' else 0)
    print('head of clinical data: ', clinical_df.head())
    print(len(clinical_df))
    print(len(rnaseq))
    X_copy = rnaseq.copy()

    # Take expression data only if it is in the patient clinical data
    rnaseq = rnaseq[rnaseq.index.isin(clinical_df.index)]
    clinical_df = clinical_df[clinical_df.index.isin(rnaseq.index)]
    new_cols = ['gene_' + str(col) for col in rnaseq.columns.to_list()]
    rnaseq.columns = new_cols
    print('new rnaseq new shape: ', rnaseq.shape)
    print(rnaseq.head())

    # Finalize X and Y variables for downstream analysis
    Y = Y[Y.index.isin(rnaseq.index)]
    Y = Y.values.ravel()
    X = pd.concat([rnaseq, clinical_df], axis=1, join="inner")
    X = X.values

    print('--------------------')
    print('rnaseq shape:', rnaseq.shape)
    print('clinical data shape: ', clinical_df.shape)
    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)
    
    ###############################################################
    print('FEATURE SELECTION')
    
    # Select k best features from the RNA-seq data
    fs = SelectKBest(score_func=f_classif, k=75)
    # apply feature selection
    rnaseq_selected = fs.fit_transform(rnaseq, Y)
    filtered = fs.get_support()
    features = np.array(rnaseq.columns)
    
    print("All features:")
    print(features)
    
    print("Selected best 3:")
    print(features[filtered])
    
    rnaseq_selected = pd.DataFrame(rnaseq_selected)
    rnaseq_selected = rnaseq_selected.set_index(rnaseq.index)
    cols_1 = ['selected_best_rnaseq_' + str(col) for col in rnaseq_selected.columns.to_list()]
    rnaseq_selected.columns = cols_1
    print(pd.DataFrame(rnaseq_selected).head())
    print(rnaseq_selected.shape)
    
    # Select best features in the clinical annotation dataset. If want to have them all, then set k = 'all'
    fs = SelectKBest(score_func=f_classif, k=5)
    print('add features shape: ', clinical_df.shape)
    print(clinical_df.head())
    clinical_df_selected = fs.fit_transform(clinical_df, Y)
    
    filtered = fs.get_support()
    features = np.array(clinical_df.columns)
    
    print("All features:")
    print(features)
    
    print("Selected best 3:")
    print(features[filtered])
    
    
    clinical_df_selected = pd.DataFrame(clinical_df_selected)
    
    
    
    clinical_df_selected = clinical_df_selected.set_index(clinical_df.index)
    cols_2 = ['selected_best_clinical_' + str(col) for col in clinical_df_selected.columns.to_list()]
    clinical_df_selected.columns = cols_2
    print(fs.scores_)
    print(pd.DataFrame(clinical_df_selected).head())
    print(clinical_df_selected.shape)
    
    X_selected = pd.concat([rnaseq_selected, clinical_df_selected], axis=1, join='inner')
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
        clinical_df_test = clinical_df[clinical_df.index.isin(X_test.index)]
        clinical_df_train = clinical_df[clinical_df.index.isin(X_train.index)]
        print('clinical_df_test', clinical_df_test.shape)
        X_test = pd.concat([pca_X_test, clinical_df_test],axis=1, join="inner")
        X_train = pd.concat([pca_X_train, clinical_df_train],axis=1, join="inner")
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
            svm.SVC(), GaussianNB()] #, CatBoostClassifier()]

# for model in models:
#     train_model(model, X_train, X_test)


# from sklearn.model_selection import GridSearchCV
#
# param_grid_lr = {
#     'max_iter': [20, 100, 200, 500],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
#     'C' : np.logspace(-4, 4, 20),
#     'penalty': ['l1', 'l2']
# }
#
# logModel_grid = GridSearchCV(estimator=LogisticRegression(random_state=1234), param_grid=param_grid_lr,
#                              verbose=1, cv=10, n_jobs=-1, scoring="recall")
# sc_x = StandardScaler()
# X_train = sc_x.fit_transform(X_train)
# X_test = sc_x.transform(X_test)
#
# logModel_grid.fit(X_train, Y_train)
# Y_test_pred = logModel_grid.predict(X_test)
# print("Accuracy on test set: {:.3}".format(accuracy_score(Y_test, Y_test_pred)))
# print("Recall-score on test set: {:.3}".format(recall_score(Y_test, Y_test_pred)))
# print("Precision-score on test set: {:.3}".format(precision_score(Y_test, Y_test_pred)))
# print("F1-score on test set: {:.3}".format(f1_score(Y_test, Y_test_pred)))
# print(f'AUC score on test set: {roc_auc_score(Y_test, Y_test_pred)}')
#
# print(logModel_grid.best_estimator_)
#








