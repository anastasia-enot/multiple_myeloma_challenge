from preprocessing import preprocess_clinical, preprocess_expression


# Please modify the paths to the 2 datasets here :

path_to_clinical_data = ''
path_to_gene_expression_data = ''

def run_preprocess_and_train(path_to_clinical_data, path_to_gene_expression_data):
    clin_data = preprocess_clinical(path_to_clinical_data)
    expression_data = preprocess_expression(path_to_gene_expression_data)
