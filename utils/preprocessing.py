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
    clean_clin_data.index.names = ['patient']

    clinical_cols_to_keep = ['D_Age', 'D_Gender', 'CYTO_predicted_feature_01',
                             'CYTO_predicted_feature_02', 'CYTO_predicted_feature_03',
                             'CYTO_predicted_feature_12', 'CYTO_predicted_feature_13',
                             'CYTO_predicted_feature_14', 'CYTO_predicted_feature_16', 'HR_FLAG']

    print('The columns that are finally kept in the clinical dataset: ')
    print(clinical_cols_to_keep)

    clean_clin_data = clean_clin_data.loc[:, clinical_cols_to_keep]
    print(f'The shape of the clinical dataset: {clean_clin_data.shape}.')

    if not os.path.exists('../validation_data'):
        os.makedirs('../validation_data')
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

    # Keep only the selected columns from the selectKbest analysis

    expression_cols_to_keep = ['gene_21', 'gene_248', 'gene_1406', 'gene_1580', 'gene_1831', 'gene_1910',
     'gene_2063', 'gene_2334', 'gene_2767', 'gene_2949', 'gene_3078', 'gene_3306',
     'gene_3760', 'gene_3853', 'gene_5275', 'gene_5744', 'gene_6915', 'gene_6953',
     'gene_7852', 'gene_8411', 'gene_9148', 'gene_9578', 'gene_10655', 'gene_11136',
     'gene_11248', 'gene_26786', 'gene_51667', 'gene_55196', 'gene_55711',
     'gene_56107', 'gene_56155', 'gene_57408', 'gene_57716', 'gene_80320',
     'gene_84889', 'gene_85462', 'gene_114827', 'gene_143502', 'gene_144817',
     'gene_145447', 'gene_150297', 'gene_162540', 'gene_162966', 'gene_255104',
     'gene_266722', 'gene_284254', 'gene_339400', 'gene_378938', 'gene_387715',
     'gene_388646', 'gene_400410', 'gene_401492', 'gene_406994', 'gene_407046',
     'gene_414926', 'gene_646268', 'gene_692093', 'gene_692233', 'gene_693206',
     'gene_693220', 'gene_728673', 'gene_100033435', 'gene_100126330',
     'gene_100132529', 'gene_100132979', 'gene_100419170', 'gene_100422875',
     'gene_100874278', 'gene_101927953', 'gene_101928376', 'gene_101929080',
     'gene_101929240', 'gene_102723775', 'gene_102724018', 'gene_102724097',
     'gene_102725072', 'gene_104326189', 'gene_105377213', 'gene_106660610',
     'gene_107985210']

    print('The columns that are finally kept in the expression dataset: ')
    print(expression_cols_to_keep)

    rna_only_first = rna_only_first.loc[:, expression_cols_to_keep]
    print(f'The shape of the expression dataset: {rna_only_first.shape}.')

    if not os.path.exists('../validation_data'):
        # Create a new directory because it does not exist
        os.makedirs('../validation_data')
        print("The folder for validation data is created: validation_data")

    rna_only_first.to_csv('validation_data/VALIDATION_BM_first_visit_all_genes.csv')
    print(f'Saved the preprocessed gene expression data to folder: validation_data/VALIDATION_BM_first_visit_all_genes.csv')
    print('--------------------------------------------------------')

    return rna_only_first

def combine_dataset(expression_dataset, clinical_dataset):
    expression_dataset = expression_dataset[expression_dataset.index.isin(clinical_dataset.index)]
    clinical_dataset = clinical_dataset[clinical_dataset.index.isin(expression_dataset.index)]
    combined = pd.concat([expression_dataset, clinical_dataset], axis=1, join='inner')

    combined.to_csv('validation_data/VALIDATION_combined_BM_first_visit_all_genes.csv')
    print(
        f'Saved the preprocessed combined data to folder: validation_data/VALIDATION_combined_BM_first_visit_all_genes.csv')

    return combined


def run_combine_datasets(path_clinical, path_expression):
    clin = preprocess_clinical(path_clinical)
    exp = preprocess_expression(path_expression)
    combined_dataset = combine_dataset(exp, clin)
    print('--------------------------------------------------------')

    return combined_dataset

if __name__ == "__main__":
    run_combine_datasets('../data/sc3_Training_ClinAnnotations.csv',
                         'data/MMRF_CoMMpass_IA9_E74GTF_Salmon_entrezID_TPM_hg19.csv')


