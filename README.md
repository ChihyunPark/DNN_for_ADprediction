# Prediction of Alzheimer's Disease Based on Deep Neural Network by Integrating Gene Expression and DNA Methylation Dataset

We developed gene expression and DNA methylation based Alzheimer's disease prediction algorithm using deep neural network.

## Website
[https://github.com/ChihyunPark/ADprediction](https://github.com/ChihyunPark/ADprediction)

## Overall algorithm


## Requirements
Python 3.5 
- numpy (1.16.3)
- pandas (0.24.2)
- scikit-learn (0.21.0)
- Tensorflow (1.4.1)

## Input and Output
### input files
- gene expression: allforDNN_ge.txt
- DNA methylation: allforDNN_me.txt
- DEG list: DEG_list.tsv
- DMP list: DMP_list.tsv
- platform of methylation data: GPL13534-11288.txt

## Codes
1. data preprocessing
	- Split_Inputdata.py
2. feature selection
	- 01 investigate_DEG_all data.R
	- 02 Annotate_DMP.py
3. hyperparameter search
	- BayesianOpt_HpParm_Search.py
4. prediction
	- AD_Prediction_ML.py
	- AD_Prediction_DNN.py

