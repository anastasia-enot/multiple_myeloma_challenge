import pandas as pd

clin_dict = pd.read_csv('data/Harmonized_Clinical_Dictionary.csv')
rnaseq = pd.read_csv('data/MMRF_CoMMpass_IA9_E74GTF_Salmon_entrezID_TPM_hg19.csv.crdownload')
clin_data = pd.read_csv('data/sc3_Training_ClinAnnotations.csv')

print(clin_dict.head())
print()
print('----------')

print(clin_data.head())

print('----------')

print(rnaseq.head())