# multiple_myeloma_challenge

The goal of this project was to train a model that will predict high-risk Multiple Myeloma patients from
the MMRF CoMMpass challenge: gene expression and clinical data. 

## How to run this project:
Recommended version: `python 3.10`

To run this project:
1. clone the repo `git clone https://github.com/anastasia-enot/multiple_myeloma_challenge.git`
2. make sure you are in the `multiple_myeloma_challenge` directory
3. Install requirements: `pip install -r requirements.txt`
4. Open the `run_test_validation_data.py` script. Please change the paths to your data (gene expression and clinical datasets). 
Do not change the paths to the model and the scaler, as they should already be present in the folder `trained_models`.
Once you have changed the paths to your data, you can just run the script. It will output the metrics (accuracy, recall, 
precision, f1 score and AUC score) and a confusion matrix. 

**Input**: expression data and clinical data

**Output**: metrics scores and a confusion matrix

## Other files
You can find the code that served to explore and preprocess the data in the notebook `notebook_preprocessing_datasets.ipynb`
The code that was used to select features and train the model can be found in the script `train_models.py`




