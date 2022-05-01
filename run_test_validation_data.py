from multiple_myeloma_challenge.utils.preprocessing import run_combine_datasets
from multiple_myeloma_challenge.utils.make_predictions_validation_data import get_predictions

# Please modify the paths to the 2 datasets here :

path_to_clinical_data = ''
path_to_gene_expression_data = ''
path_to_model = 'trained_models/LogReg_model_BM_80_9_first_visit.sav'
path_to_scaler = 'trained_models/standard_scaler_BM_80_9_first_visit.sav'

combined_dataset = run_combine_datasets(path_clinical=path_to_clinical_data, path_expression=path_to_gene_expression_data)
print(combined_dataset)
get_predictions(path_to_model, path_to_scaler, combined_dataset)




    
