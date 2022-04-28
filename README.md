# multiple_myeloma_challenge

## How to run the script on validation data



## How the model was trained

### 1. preprocessing step
The goal of this step is to prepare the data for ML. In the clinical data file we check every column,
and remove columns that are unnecessary.
- In the expression data, there is need to identify different samples. There is PB = peripheral blood.
As there is not enough data on this, we remove it. Then there is BM = bone marrow. Some patients come 
several times. For this analysis we take only the information from the first visit. This is done to
simplify the analysis as not all the patients have the same number of visits. 
- the data was tested as is, with no transformation, and the Logistic Regression gave the best baseline ... PUT
BASELINE 
- Tested PCA, selectKbest and different models (SVM, Random Forest, XGBoost, Catboost)
- surprsingly among all the models, logistic Regression performed best.
- Cross validation
- critique: we could select the best genes by other techniques ; include other patient data such as data on the treatments received. 
- 
