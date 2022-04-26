# multiple_myeloma_challenge

## How to run the script


### 1. preprocessing step
The goal of this step is to prepare the data for ML. In the clinical data file we check every column,
and remove columns that are unnecessary.
In the expression data, there is need to identify different samples. There is PB = peripheral blood.
As there is not enough data on this, we remove it. Then there is BM = bone marrow. Some patients come 
several times. For this analysis we take only the information from the last visit. This is done to
simplify the analysis as not all the patients have the same number of visits. 
