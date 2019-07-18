# Prediction of Alzheimer's Disease Based on Deep Neural Network by Integrating Gene Expression and DNA Methylation Dataset

We developed gene expression and DNA methylation based Alzheimer's disease prediction algorithm using deep neural network.

## Website
https://github.com/ChihyunPark/DNN_for_ADprediction

## Overall algorithm
![github_f1](https://user-images.githubusercontent.com/34843393/61431751-8801f280-a969-11e9-87b3-aa3d2b569abc.PNG)

## Requirements
Python 3.5 
- numpy (1.16.3)
- pandas (0.24.2)
- scikit-learn (0.21.0)
- Tensorflow (1.4.1)

## Input and Output
### input files
- gene expression: allforDNN_ge_sample.txt
- DNA methylation: allforDNN_me_sample.txt

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

