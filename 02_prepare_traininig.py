import pandas as pd
from sklearn.model_selection import train_test_split

rnaseq = pd.read_csv('data/MMRF_CoMMpass_IA9_E74GTF_Salmon_entrezID_TPM_hg19.csv')
rnaseq = rnaseq.drop(['Unnamed: 0'],axis=1)
new = []
for col in rnaseq.columns:
    new_c = col.replace('_1_BM', '')
    new_c = new_c.replace('_2_BM', '')
    new.append(new_c)
rnaseq.columns = new
col_list = rnaseq.columns.to_list()
print(len(col_list))
col_list = list(set(col_list))
print(len(col_list))
rnaseq = rnaseq.T
print(rnaseq.head())

print(rnaseq.isnull())

print('------------')
print(rnaseq.head())

Y = pd.read_csv('data/clean_clin_data.csv')
Y = Y.set_index('Patient')
print(Y.head())

rnaseq['HR_FLAG'] = Y['HR_FLAG']
print(rnaseq['HR_FLAG'].value_counts())
rnaseq = rnaseq.drop(rnaseq[rnaseq['HR_FLAG'] == 'CENSORED'].index)
print('------------')
print(rnaseq['HR_FLAG'].value_counts())

X = rnaseq.drop(['HR_FLAG'],axis=1)

Y = rnaseq.loc[:, 'HR_FLAG']
print(Y.isnull())
print('----------')
#print(Y.loc[:, Y.isna().any()])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)