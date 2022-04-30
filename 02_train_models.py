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
import itertools


def prepare_data_for_train(path_expression, path_clinical, num_feat_expression, num_feat_clinical):
    '''
    This function will prepare the data for the training step. The data should already be 'clean': it should contain 
    only Bone Marrow (BM) patients from their 1st visit. (samples ending with 1_BM).
    :param path_expression: (str) the path to expression data. 
    :param path_clinical: (str) the path to clinical data
    :param num_feat_expression: (int) number of
    :return: 
    '''
    
    rnaseq = pd.read_csv(path_expression, index_col='patient')
    clinical_df = pd.read_csv(path_clinical, index_col='Patient')

    #clinical_df = clinical_df.set_index('Patient')
    print('Clinical data shape: ', clinical_df.shape)
    Y = clinical_df[['HR_FLAG']]
    
    clinical_df = clinical_df.drop('HR_FLAG', axis=1)
    print(f'Length of the clinical dataset: {len(clinical_df)}.')
    print(f'Length of the expression dataset: {len(rnaseq)}.')
    #X_copy = rnaseq.copy()

    # Take expression data only if it is in the patient clinical data and vice versa
    rnaseq = rnaseq[rnaseq.index.isin(clinical_df.index)]
    clinical_df = clinical_df[clinical_df.index.isin(rnaseq.index)]

    print(f'Length of the clinical dataset after checking the overlapping patients: {len(clinical_df)}.')
    print(f'Length of the expression dataset after checking the overlapping patients: {len(rnaseq)}.')


    # Finalize X and Y variables for downstream analysis
    Y = Y[Y.index.isin(rnaseq.index)]
    Y = Y.values.ravel()

    
    ###############################################################

    select_best = False
    if select_best == True:
        print('FEATURE SELECTION')

        # Select k best features from the RNA-seq data
        fs = SelectKBest(score_func=f_classif, k=num_feat_expression)
        # apply feature selection
        rnaseq_selected = fs.fit_transform(rnaseq, Y)
        filtered = fs.get_support()
        features = np.array(rnaseq.columns)


        print(f"Selected best {num_feat_expression} for expression data:")
        print(features[filtered])

        rnaseq_selected = pd.DataFrame(rnaseq_selected)
        rnaseq_selected = rnaseq_selected.set_index(rnaseq.index)
        cols_1 = ['selected_best_rnaseq_' + str(col) for col in rnaseq_selected.columns.to_list()]
        rnaseq_selected.columns = cols_1
        print(pd.DataFrame(rnaseq_selected).head())
        print(rnaseq_selected.shape)
    
        # Select best features in the clinical annotation dataset. If want to have them all, then set k = 'all'
        fs = SelectKBest(score_func=f_classif, k=num_feat_clinical)
        print('add features shape: ', clinical_df.shape)
        print(clinical_df.head())
        clinical_df_selected = fs.fit_transform(clinical_df, Y)

        filtered = fs.get_support()
        features = np.array(clinical_df.columns)

        print(f"Selected best {num_feat_clinical} for clinical data:")
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
        X = X_selected

    else:
        X = pd.concat([rnaseq, clinical_df], axis=1, join="inner")
        X = X.values

    print('--------------------')
    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)
    print('--------------------')

    # apply_PCA = False
    # if apply_PCA == True:
    #     print('AFTER PCA')
    #     from sklearn.decomposition import PCA
    #     pca = PCA(random_state=20)
    #     pca.fit(X_copy)
    #     explained_variance = np.cumsum(pca.explained_variance_ratio_)
    #     # plt.vlines(x=80, ymax=1, ymin=0, colors="r", linestyles="--")
    #     # plt.hlines(y=0.95, xmax=120, xmin=0, colors="g", linestyles="--")
    #     plt.plot(explained_variance)
    #     plt.show()
    #
    #     # We need 13 PC to explain 95% of variance
    #     pca = PCA(n_components=500)
    #     pca_X = pca.fit_transform(X_copy)
    #     pca_X = pd.DataFrame(pca_X)
    #     pca_X = pca_X.set_index(X_copy.index)
    #     pca_X_test = pca_X[pca_X.index.isin(X_test.index)]
    #     print('pca_X_test', pca_X_test.shape)
    #     pca_X_train = pca_X[pca_X.index.isin(X_train.index)]
    #     clinical_df_test = clinical_df[clinical_df.index.isin(X_test.index)]
    #     clinical_df_train = clinical_df[clinical_df.index.isin(X_train.index)]
    #     print('clinical_df_test', clinical_df_test.shape)
    #     X_test = pd.concat([pca_X_test, clinical_df_test], axis=1, join="inner")
    #     X_train = pd.concat([pca_X_train, clinical_df_train], axis=1, join="inner")
    #     print(X_test.shape)
    #     print(X_train.shape)
    #     # pca_X = pca.transform(X_test)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, stratify = Y, random_state=42)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_train shape: {X_test.shape}')
    print(f'X_train shape: {Y_train.shape}')
    print(f'X_train shape: {Y_test.shape}')
    print('-------------------------')

    return X_train, X_test, Y_train, Y_test


def train_model(model, X_train, X_test, Y_train, Y_test):
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

    return recall_score(Y_test, Y_test_pred), accuracy_score(Y_test, Y_test_pred)

models = [LogisticRegression()] #, xgb.XGBClassifier(use_label_encoder=False, n_jobs=7), RandomForestClassifier(),
            #svm.SVC(), GaussianNB()] #, CatBoostClassifier()]

list_expression = list(range(50, 350, 15))
print(list_expression[:10])
list_clinical = list(range(1, 15, 2))
num_features = []
for ex in list_expression:
    for cl in list_clinical:
        num_features.append((ex, cl))

print(num_features[:10])
path_expression = 'data/BM_first_visit_all_genes.csv'
path_clinical = 'data/clean_clinical_annotations.csv'
df = pd.DataFrame()
i = 0
from tqdm import tqdm
for model in models:
    for tup in tqdm(num_features):
        X_train, X_test, Y_train, Y_test = prepare_data_for_train(path_expression, path_clinical,
                                                              num_feat_expression=tup[0], num_feat_clinical=tup[1])
        # print(tup)
        # rec, acc = train_model(model, X_train, X_test, Y_train, Y_test)
        # df.loc[i, 'tup'] = str(tup)
        # df.loc[i, 'accuracy'] = acc
        # df.loc[i, 'recall'] = rec
        # i = i + 1
        #
        # print('---------------------')
    #df.to_csv('data/logreg_Kbest_selection.csv')


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







