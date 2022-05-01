import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_predictions(path_to_model, path_to_scaler, combined_dataset):
    loaded_model = pickle.load(open(path_to_model, 'rb'))

    X_test = combined_dataset.drop(['HR_FLAG'], axis=1)
    X_test = X_test.values

    # Normalize data
    sc_x = pickle.load(open(path_to_scaler, 'rb'))
    X_test = sc_x.transform(X_test)

    # Define the target
    Y_test = combined_dataset.loc[:, 'HR_FLAG']
    Y_test = Y_test.values.ravel()

    Y_test_pred = loaded_model.predict(X_test)

    # Compute metrics
    print('-----------------------')
    print(f'Metrics for the model {str(loaded_model)}: ')
    print("Accuracy on test set: {:.3}".format(accuracy_score(Y_test, Y_test_pred)))
    print("Recall-score on test set: {:.3}".format(recall_score(Y_test, Y_test_pred)))
    print("Precision-score on test set: {:.3}".format(precision_score(Y_test, Y_test_pred)))
    print("F1-score on test set: {:.3}".format(f1_score(Y_test, Y_test_pred)))
    print(f'AUC score on test set: {roc_auc_score(Y_test, Y_test_pred)}')
    print('-----------------------')

    cm = confusion_matrix(Y_test, Y_test_pred)
    sns.heatmap(cm , annot=True , fmt='d')
    plt.show()