import pandas as pd
import os

def preprocess_clinical(path):
    '''

    :param path: (str) path to the clinical dataset
    :return: the cleaned preprocessed dataset ready to be used for the model prediction
    '''
    print('Started preprocessing of the clinical dataset...')
    clin_data = pd.read_csv(path)
    clin_data.head()

    # I remove the patients with CENSORED HR_FLAG as I will not perform survival analysis
    clin_data = clin_data.drop(clin_data[clin_data['HR_FLAG'] == 'CENSORED'].index)
    clin_data['HR_FLAG'] = clin_data['HR_FLAG'].apply(lambda x: 1 if x == 'TRUE' else 0)

    # Remove all the columns that contain only NA values
    clin_data = clin_data.dropna(axis=1, how='all')
    print('The description of the clinical dataset: ')
    print(clin_data.describe())

    # Show all the columns that are in this dataset to proceed to manual analysis
    print(f'The columns in the clinical dataset: {clin_data.columns}')

    # Remove columns that are not useful or redundant
    # Patient type, 'RNASeq_transLevelExpFile', 'RNASeq_geneLevelExpFile' is always the same
    # 'RNASeq_transLevelExpFileSamplId', 'RNASeq_geneLevelExpFileSamplId' is the id, unique per patient
    # The D_ISS columns corresponds to the stage of the disease, however some of the patients contain NaN values. Due to the lack of time,
    # I did not go into imputation techniques to handle those missing values, therefore I removed the D_ISS column

    cols_to_del = ['Study', 'D_OS', 'D_OS_FLAG', 'D_PFS',
                   'D_PFS_FLAG', 'D_ISS', 'PatientType', 'RNASeq_transLevelExpFile', 'RNASeq_transLevelExpFileSamplId',
                   'RNASeq_geneLevelExpFile', 'RNASeq_geneLevelExpFileSamplId', 'WES_mutationFileMutect',
                   'WES_mutationFileStrelkaSNV', 'WES_mutationFileStrelkaIndel', 'WES_mutationFileStrelkaIndel']

    print(f'The columns that were deleted from the clinical dataset: {cols_to_del}')

    clean_clin_data = clin_data.drop(cols_to_del, axis=1)

    # The column CYTO_predicted_feature_10 contains NaN, so due to the lack of time,
    # I did not go into imputation techniques to handle those missing values, therefore I removed the CYTO_predicted_feature_10 column
    print('The additional column to be deleted is: CYTO_predicted_feature_10')
    clean_clin_data = clean_clin_data.drop(['CYTO_predicted_feature_10'], axis=1)

    # Binarize the gender column
    clean_clin_data['D_Gender'] = clean_clin_data['D_Gender'].apply(lambda x: 1 if x == 'Female' else 0)

    clean_clin_data = clean_clin_data.set_index('Patient')

    if not os.path.exists('validation_data'):
        # Create a new directory because it does not exist
        os.makedirs('validation_data')
        print("The folder for validation data is created: validation_data")

    clean_clin_data.to_csv('validation_data/VALIDATION_clean_clinical_annotations.csv')
    print('Saved the cleaned validation clinical dataset to the folder: validation_data/VALIDATION_clean_clinical_annotations.csv')
    print('--------------------------------------------------------')

    return clean_clin_data


def preprocess_expression(path):
    # Define options on which dataset to save, all genes or selected, last or first visit
    # Here, it will probably be only the first visit all genes

    print('Started preprocessing of the data expression dataset...')
    rna = pd.read_csv(path)

    rna = rna.set_index('Unnamed: 0')
    rna.index.names = ['gene']
    rna = rna.rename(index=lambda x: 'gene_' + str(x))

    list_cols = rna.columns.to_list()

    only_BM_cols = []
    for col in list_cols:
        spl = col.split('_')[-1]
        if spl == 'BM':
            only_BM_cols.append(col)

    rna_only_BM = rna.loc[:, only_BM_cols]
    print('Selected only Bone Marrow samples for the gene expression dataset.')

    # take only the first visit for patients
    only_first_visit_cols = []
    patient_numbers = []
    for col in only_BM_cols:
        pat = '_'.join(col.split('_')[:2])
        visit = col.split('_')[2]
        if visit == '1':
            only_first_visit_cols.append(col)
            patient_numbers.append(pat)

    rna_only_first = rna_only_BM.loc[:, only_first_visit_cols]
    print('Selected only the first visit for each patient.')

    renamed_cols = []
    for col in rna_only_first.columns:
        pat = '_'.join(col.split('_', 2)[0:2])
        renamed_cols.append(pat)

    rna_only_first.columns = renamed_cols
    rna_only_first = rna_only_first.T
    rna_only_first.index.names = ['patient']

    if not os.path.exists('validation_data'):
        # Create a new directory because it does not exist
        os.makedirs('validation_data')
        print("The folder for validation data is created: validation_data")

    rna_only_first.to_csv('validation_data/VALIDATION_BM_first_visit_all_genes.csv')
    print(f'Saveed the preprocessed gene expression data to folder: validation_data/VALIDATION_BM_first_visit_all_genes.csv')

