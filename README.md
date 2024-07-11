
# Project structure

```
thesis-project
└───data_processing
│       │   al_data_selection.ipynb
│       │   combine_test_data.ipynb
│       │   inspect_data.ipynb
│       │   preprocessing.ipynb
└───eval
│       │   evaluation_error_export.ipynb
└───fine-tuning
        │   predict.py
        │   train_model.py
│   .gitignore
│   LICENSE
│   README.md
│   requirements.tx
```

## Code
### 1. Data Processing
- preprocessing.ipynb :
The raw unstructured data consist of Electronic Health Records (EHR) represented in csv format which contains various kinds of information about the record. The classifiers used in this project has been trained and fine-tuned with anonymized sentences. Therefore the unstructured data had to be filtered from unnnecessary information, text had to be segmented into sentences and person and location entities had to be anonymized. The code in this notebook includes the steps and the code for this process.  
  

- inspect_data.ipynb :
A generic data analysis code used in this research, calculates category distribution, pattern distribution, picking and comparing individual instances 


- al_data_selection.ipynb : 
This notebook includes the code that has been used in data querying within the Active Learning (AL) scheme. 
  - Uncertainty based selection: The code that samples instances based on defined confidence range.
  - Cosine Similarity based selection: The code for representing the sentences in word embedding averages and applying DBSCAN clustering aiming to get most informative representative instances.


- combine_test_data.ipynb :
This project combined 3 independent test sets to create a more balanced test data. This code demonstrates the process of combining these datasets.
### 2. Fine-tuning
These scripts are implemented directly from the previous A-Proof research repository. train_model.py is used to fine-tune a pre-trained model, predict.py is used to create predictions and output the updated evaluation data. They both use simpletransformers python library. 

- train_model.py : This script used to fine-tune a pre-trained model. It needs access to a model folder and training hyperparameter can be defined within the script. It outputs the new fine-tuned model into the defined path.


- predict.py : This script is used to create predictions from a model for a given evaluation/test data. The data that model predicts over needs to be in a dataframe stored in pickle and text instances should be under a column named as 'text' within the dataframe. It takes the pickled dataframe as input, adds new columns for classification and confidence values and writes out the dataframe as pickle over the original input.
### 3. Evaluation
- evaluation_error_export.ipynb :
This notebook includes the code for evaluation of predictions. First part of the notebook it calculates the evaluation metrics of precision, recall and creates a classification report. Then, in the last part it creates confusion matrices for the given evaluation. 
## Thesis report

The full thesis can be found in this folder.

## Data 
The data used in this thesis consist of Electronic Health Records (EHR) from Amsterdam University Medical Centers (AUMC) and VU Medical Center (VUMC). Due to privacy and confidentiality agreements, these datasets are not publicly accessible.








