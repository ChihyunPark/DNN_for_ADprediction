####################################################################################################################################################
# BayesianOpt_HpParm_Search.py
# Author: Chihyun Park
# Email: chihyun.park@yonsei.ac.kr
# Date created: 06/13/2019
# Date lastly modified: 7/1/2019
# Purpose: To perform hyperparameter search using Bayesian optimization
# Reference: https://github.com/huseinzol05/Deep-Learning-Tensorflow/tree/master/Bayesian-Optimization
# input
#       1. gene expression (samples x genes) with label (AD, Normal)
#       2. DNA methylation (samples x CpG probes) with label (AD, Normal)
#       3. DEG list (Normal vs AD)
#       4. DMP list (Normal vs AD)
# output
#       best combination of hyper-parameters
####################################################################################################################################################


import pandas as pd
import tensorflow as tf
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import os
import random as rd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

## divide data into X and Y
def partitionTrainTest_ML_for_CV_DNN(xy_me_ge_values):
	np.random.shuffle(xy_me_ge_values)

	x_data_List = []
	y_data_List = []

	colSize = len(xy_me_ge_values[0])

	for i in range(len(xy_me_ge_values)):
		x_tmpRow = xy_me_ge_values[i, 0:colSize - 2]
		y_tmpRow = xy_me_ge_values[i, colSize - 1:colSize]

		x_data_List.append(x_tmpRow)
		y_data_List.append(y_tmpRow)

	return np.array(x_data_List), np.array(y_data_List)


## do machine learning (RF, SVM, NB) from the given training and test dataset
def doMachineLearning_single_Kfold(xy_train, xy_test, outfilename, k):
	print("\n\ndoMachineLearning_single")

	# divide data into X and Y
	x_train, y_train = partitionTrainTest_ML_for_CV_DNN(xy_train)
	x_test, y_test = partitionTrainTest_ML_for_CV_DNN(xy_test)

	y_train = y_train.astype(np.int)
	y_test = y_test.astype(np.int)

	print("x_train: " + str(x_train.shape))
	print("x_test: " + str(x_test.shape))
	print("y_train: " + str(y_train.shape))
	print("y_test: " + str(y_test.shape))

	with open(outfilename, 'w') as fout:
		fout.write("Do DNN with DNA methylation and Gene expression\n")

		###############################################################################################
		# define RandomForest
		print("## Random Forest")
		fout.write("\n\n")
		fout.write("Random Forest")
		rdf_clf = RandomForestClassifier(criterion='entropy', oob_score=True, n_estimators=100, n_jobs=-1,
										 random_state=0, max_depth=6)

		rdf_clf.fit(x_train, y_train)
		predicted = rdf_clf.predict(x_test)
		test_acc = accuracy_score(y_test, predicted)
		training_acc = accuracy_score(y_train, rdf_clf.predict(x_train))
		roc_auc = metrics.roc_auc_score(y_test.ravel(), predicted.ravel())

		print(str(k) + "-fold training accuracy:" + "\t" + str(training_acc))
		print(str(k) + "-fold test accuracy:" + "\t" + str(test_acc))
		print(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))

		fout.write(str(k) + "-fold training accuracy:" + "\t" + str(training_acc) + "\n")
		fout.write(str(k) + "-fold test accuracy:" + "\t" + str(test_acc) + "\n")
		fout.write(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))

		###############################################################################################
		# define SVM
		print("## SVM")
		fout.write("\n\n")
		fout.write("SVM (with RBF)" + "\n")

		svm_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
						  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
						  max_iter=-1, probability=True, random_state=None, shrinking=True,
						  tol=0.001, verbose=False)
		svm_clf.fit(x_train, y_train)

		test_predictions = svm_clf.predict(x_test)
		test_predictions = np.reshape(test_predictions, (-1, 1))
		training_acc = accuracy_score(y_train, svm_clf.predict(x_train))
		test_acc = accuracy_score(y_test, test_predictions)
		roc_auc = metrics.roc_auc_score(y_test, test_predictions)

		print(str(k) + "-fold training accuracy:" + "\t" + str(training_acc))
		print(str(k) + "-fold test accuracy:" + "\t" + str(test_acc))
		print(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))

		fout.write(str(k) + "-fold training accuracy:" + "\t" + str(training_acc) + "\n")
		fout.write(str(k) + "-fold test accuracy:" + "\t" + str(test_acc) + "\n")
		fout.write(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))

		###############################################################################################
		# define naive bayesian classifier
		print("## naive bayesian")
		fout.write("\n\n")
		fout.write("naive bayesian" + "\n")
		###############################################################################################
		# define a model #rdf_clf.fit(x_data_std, y_data)
		gnb_clf = GaussianNB()
		gnb_clf.fit(x_train, y_train)

		test_predictions = gnb_clf.predict(x_test)
		test_predictions = np.reshape(test_predictions, (-1, 1))
		training_acc = accuracy_score(y_train, gnb_clf.predict(x_train))
		test_acc = accuracy_score(y_test, test_predictions)
		roc_auc = metrics.roc_auc_score(y_test, test_predictions)

		print(str(k) + "-fold training accuracy:" + "\t" + str(training_acc))
		print(str(k) + "-fold test accuracy:" + "\t" + str(test_acc))
		print(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))

		fout.write(str(k) + "-fold training accuracy:" + "\t" + str(training_acc) + "\n")
		fout.write(str(k) + "-fold test accuracy:" + "\t" + str(test_acc) + "\n")
		fout.write(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))

	fout.close()


## initialization with xavier
def xavier_init(n_inputs, n_outputs, uniform=True):
	if uniform:
		init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
		return tf.random_uniform_initializer(-init_range, init_range)
	else:
		stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
	return tf.truncated_normal_initializer(stddev=stddev)


## get important features in gene expression
def getDEG_limma(filename, Thres_lfc, Thres_pval):
	geneSet = set()
	f = open(filename, 'r')
	inCSV = csv.reader(f, delimiter="\t")
	header = next(inCSV)  # for header

	for row in inCSV:
		gene = row[0]
		logFC = float(row[1])
		Pval = float(row[4]) ## adj p-val : row[5]

		if abs(logFC) >= Thres_lfc and Pval < Thres_pval:
			geneSet.add(gene)

	print("[limma] Number of gene set: " + str(len(geneSet)))

	return geneSet


## get important features in methylation
def getDMG(filename):
	geneSet = set()
	cpgSet = set()

	f = open(filename, 'r')
	inCSV = csv.reader(f, delimiter="\t")
	header = next(inCSV)  # for header
	for row in inCSV:
		cpg = row[0]
		gene = row[6]
		cpgSet.add(cpg)
		geneSet.add(gene)

	print("Number of CpG set: " + str(len(cpgSet)))
	print("Number of gene set: " + str(len(geneSet)))
	return geneSet, cpgSet


## reduce the dimension from raw gene expression dataset
def applyDimReduction_DEG_intersectGene(infilename, geneSet, filter_fn, Thres_lfc, Thres_pval):
	print("applyDimReduction_DEG_intersectGene")

	selected_genelist = ['SampleID']
	f = open(filter_fn, 'r')
	inCSV = csv.reader(f, delimiter="\t")
	header = next(inCSV)  # for header

	gene_fc_dict = {}
	for row in inCSV:
		gene = row[0]
		logFC = float(row[1])
		Pval = float(row[4])  ## adj p-val : row[5]

		if abs(logFC) >= Thres_lfc and Pval < Thres_pval:
			if gene in geneSet:
				gene_fc_dict[gene] = logFC


	sorted_gene_fc_list = sorted(gene_fc_dict.items(), key=lambda x: x[1], reverse=True)
	print(sorted_gene_fc_list)

	cnt = 0
	for k_v in sorted_gene_fc_list:
		key = k_v[0]
		selected_genelist.append(key)

	# Label_No	Label_AD
	selected_genelist.append('Label_No')
	selected_genelist.append('Label_AD')

	print(str(len(selected_genelist)))
	print(selected_genelist)

	xy_all_df = pd.read_csv(infilename, sep='\t')
	xy_sel_df = xy_all_df[selected_genelist]
	xy = xy_sel_df.as_matrix()
	print("xy shape: " + xy.shape.__str__())

	xy_values = xy[:, 1:-2]
	xy_labels = xy[:, -2:]

	print(xy_values)
	print(xy_labels)

	# Label transformation: one hot | [1 0], [0 1] = No, AD --> one column | 0 or 1 = No, AD
	xy_labels_1_column = []

	NoArr = [1, 0]
	# AD array
	ADArr = [0, 1]
	num_rows, num_cols = xy_labels.shape
	for i in range(num_rows):
		if np.array_equal(xy_labels[i], NoArr):
			xy_labels_1_column.append(0)
		if np.array_equal(xy_labels[i], ADArr):
			xy_labels_1_column.append(1)

	print("xy_values row: " + len(xy_values).__str__() + "\t" + " col: " + len(xy_values[0]).__str__())
	print("xy_labels row: " + len(xy_labels).__str__() + "\t" + " col: " + len(xy_labels[0]).__str__())

	X_embedded = xy_values
	print(X_embedded.shape)
	XY_embedded = np.append(X_embedded, xy_labels, axis=1)
	print(XY_embedded.shape)
	print(XY_embedded)
	return XY_embedded


## reduce the dimension from raw methylation dataset
def applyDimReduction_DMP_intersectGene(infilename, geneSet, filter_fn):
	print("applyDimReduction_DMP_intersectGene")
	selected_cpglist = ['SampleID']
	f = open(filter_fn, 'r')
	inCSV = csv.reader(f, delimiter="\t")

	for row in inCSV:
		cpg = row[0]
		gene = row[6]
		if gene in geneSet:
			selected_cpglist.append(cpg)

	selected_cpglist.append('Label_No')
	selected_cpglist.append('Label_AD')
	print(str(len(selected_cpglist))) ## SampleID, CpG,,,, Label_NO, Label_AD
	print(selected_cpglist) ## 143 features

	#xy_all_df = pd.read_csv(infilename, sep='\t')
	#xy_sel_df = xy_all_df[selected_cpglist]

	xy_sel_df = pd.read_csv(infilename, sep='\t', usecols=selected_cpglist)
	xy = xy_sel_df.as_matrix() ## sampleID, expr + label 2 columns
	print("xy shape: " + xy.shape.__str__())
	print(xy)

	xy_tp = np.transpose(xy)
	print("xy_tp shape: " + xy_tp.shape.__str__())

	xy_values = xy[:, 1:-2]
	xy_labels = xy[:, -2:]

	#print(xy_values)
	#print(xy_values.shape)
	#print(xy_labels)
	#print(xy_labels.shape)

	# Label transformation: one hot | [1 0], [0 1] = No, AD --> one column | 0 or 1 = No, AD
	xy_labels_1_column = []

	NoArr = [1, 0]
	# AD array
	ADArr = [0, 1]
	num_rows, num_cols = xy_labels.shape
	for i in range(num_rows):
		if np.array_equal(xy_labels[i], NoArr):
			xy_labels_1_column.append(0)
		if np.array_equal(xy_labels[i], ADArr):
			xy_labels_1_column.append(1)

	print("xy_tp row: " + len(xy_tp).__str__() + "\t" + " col: " + len(xy_tp[0]).__str__())
	print("xy_values row: " + len(xy_values).__str__() + "\t" + " col: " + len(xy_values[0]).__str__())
	print("xy_labels row: " + len(xy_labels).__str__() + "\t" + " col: " + len(xy_labels[0]).__str__())

	X_embedded = xy_values
	print(X_embedded.shape)
	XY_embedded = np.append(X_embedded, xy_labels, axis=1)
	print(XY_embedded.shape)
	print(XY_embedded)

	return XY_embedded


## integrated multi-omcis dataset
def buildIntegratedDataset_DNN(xy_gxpr, xy_meth, mode):
	print("buildIntegratedDataset")
	xy_gxpr_meth = []

	n_row_g, n_col_g = xy_gxpr.shape
	n_row_m, n_col_m = xy_meth.shape

	# build random index pair set
	idxSet_No = set()
	idxSet_AD = set()

	NoArr = [1, 0]
	ADArr = [0, 1]
	NoCnt = 0
	ADCnt = 0

	for idx_g in range(0, n_row_g - 1):
		label_g = xy_gxpr[idx_g][-2:]
		# print(label_g)
		for idx_m in range(0, n_row_m - 1):
			label_m = xy_meth[idx_m][-2:]
			# print(label_m)

			# normal
			if np.array_equal(label_g, NoArr) and np.array_equal(label_m, NoArr):
				integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
				idxSet_No.add(integ_idx)
				# print("normal: " + integ_idx)
				NoCnt += 1

			# AD
			if np.array_equal(label_g, ADArr) and np.array_equal(label_m, ADArr):
				integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
				idxSet_AD.add(integ_idx)
				# print("ad: " + integ_idx)
				ADCnt += 1

	print("NoCnt: " + NoCnt.__str__())
	print("ADCnt: " + ADCnt.__str__())
	print("size of idxSet_No: " + len(idxSet_No).__str__())
	print("size of idxSet_AD: " + len(idxSet_AD).__str__())

	balanced_sample_size = 0;
	if(len(idxSet_No) > len(idxSet_AD)):
		balanced_sample_size = len(idxSet_AD)

	if (len(idxSet_AD) > len(idxSet_No)):
		balanced_sample_size = len(idxSet_No)

	if mode == "balanced":
		print("balanced_sample_size: " + balanced_sample_size.__str__())

		# for normal
		cnt = 0
		for idx in range(len(idxSet_No)):
			idx_str = idxSet_No.pop()
			idx_str_split_list = idx_str.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_me)):
				xy_me_ge_values_tmp.insert(i + 1, value_me[i])

			for j in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

			#xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

			cnt += 1
			if(cnt >= balanced_sample_size):
				break

		# for AD
		cnt = 0
		for idx in range(len(idxSet_AD)):
			idx_str = idxSet_AD.pop()
			idx_str_split_list = idx_str.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_me)):
				xy_me_ge_values_tmp.insert(i + 1, value_me[i])

			for j in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

			#xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

			cnt += 1
			if (cnt >= balanced_sample_size):
				break

	if mode != "balanced":
		# for normal
		for idx in range(len(idxSet_No)):
			idx_str = idxSet_No.pop()
			idx_str_split_list = idx_str.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_me)):
				xy_me_ge_values_tmp.insert(i + 1, value_me[i])

			for j in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

			#xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

		# for AD
		for idx in range(len(idxSet_AD)):
			idx_str = idxSet_AD.pop()
			idx_str_split_list = idx_str.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_me)):
				xy_me_ge_values_tmp.insert(i + 1, value_me[i])

			for j in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_ge[j])

			#xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)


	xy_me_ge_values = np.array(xy_gxpr_meth)
	print(xy_me_ge_values.shape)

	return xy_me_ge_values


## integrated multi-omcis dataset with selected samples
def buildIntegratedDataset_DNN_selectN(xy_gxpr, xy_meth, mode, nSamples):
	print("buildIntegratedDataset")
	xy_gxpr_meth = []

	n_row_g, n_col_g = xy_gxpr.shape
	n_row_m, n_col_m = xy_meth.shape

	# build random index pair set
	idxSet_No = set()
	idxSet_AD = set()

	NoArr = [1,0]
	ADArr = [0,1]
	NoCnt = 0
	ADCnt = 0

	for idx_g in range(0, n_row_g):
		label_g = xy_gxpr[idx_g][-2:]

		for idx_m in range(0, n_row_m):
			label_m = xy_meth[idx_m][-2:]

			# normal
			if np.array_equal(label_g, NoArr) and np.array_equal(label_m, NoArr):
				integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
				idxSet_No.add(integ_idx)
				NoCnt += 1

			# AD
			if np.array_equal(label_g, ADArr) and np.array_equal(label_m, ADArr):
				integ_idx = idx_g.__str__() + "_" + idx_m.__str__()
				idxSet_AD.add(integ_idx)
				ADCnt += 1

	print("NoCnt: " + NoCnt.__str__())
	print("ADCnt: " + ADCnt.__str__())

	balanced_sample_size = 0;
	if (NoCnt > ADCnt):
		balanced_sample_size = ADCnt

	if (NoCnt < ADCnt):
		balanced_sample_size = NoCnt

	idxList_No = list(idxSet_No)
	idxList_AD = list(idxSet_AD)
	rd.shuffle(idxList_No)  ## randomly suffle the samples
	rd.shuffle(idxList_AD)  ## randomly suffle the samples

	print("idxList_No: " + str(idxList_No))
	print("idxList_AD: " + str(idxList_AD))

	if mode == "balanced":
		print("balanced mode")

		# for normal
		cnt = 0
		for idx in idxList_No:
			idx_str_split_list = idx.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(i + 1, value_ge[i])

			for j in range(len(value_me)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_me[j])

			# xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

			cnt += 1
			if (cnt >= balanced_sample_size) or (cnt >= nSamples):
				break

		# for AD
		cnt = 0
		for idx in idxList_AD:
			idx_str_split_list = idx.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(i + 1, value_ge[i])

			for j in range(len(value_me)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_me[j])

			# xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

			cnt += 1
			if (cnt >= balanced_sample_size) or (cnt >= nSamples):
				break

	if mode != "balanced":
		print("unbalanced mode")

		# for normal
		for idx in idxList_No:
			idx_str_split_list = idx.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(i + 1, value_ge[i])

			for j in range(len(value_me)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_me[j])

			# xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

		# for AD
		for idx in idxList_AD:
			idx_str_split_list = idx.split('_')

			idx_ge_str = idx_str_split_list[0]
			idx_me_str = idx_str_split_list[1]
			idx_ge = int(idx_ge_str)
			idx_me = int(idx_me_str)

			value_ge = xy_gxpr[idx_ge][:-2]
			value_me = xy_meth[idx_me][:-2]

			xy_me_ge_values_tmp = []
			xy_me_ge_values_tmp.insert(0, idx_ge_str + "_" + idx_me_str)

			for i in range(len(value_ge)):
				xy_me_ge_values_tmp.insert(i + 1, value_ge[i])

			for j in range(len(value_me)):
				xy_me_ge_values_tmp.insert(j + len(xy_me_ge_values_tmp), value_me[j])

			# xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 0)
			xy_me_ge_values_tmp.insert(len(xy_me_ge_values_tmp) + 1, 1)
			xy_gxpr_meth.append(xy_me_ge_values_tmp)

	xy_me_ge_values = np.array(xy_gxpr_meth)
	print(xy_me_ge_values.shape)

	return xy_me_ge_values


## change encoding of y
def changeEncoding(Y):
	# No array
	NoArr = ['0']
	# AD array
	ADArr = ['1']

	Y_change = []
	for label in Y:
		if np.array_equal(label, ADArr):
			Y_change.append([0, 1])
		if np.array_equal(label, NoArr):
			Y_change.append([1, 0])

	return np.array(Y_change)


## divide data into train, test
def partitionTrainTest_balanced2(xy_me_ge_values, train_test_ratio):

	## select only X
	x_value = xy_me_ge_values[:, 1:len(xy_me_ge_values[0]) - 1]

	## normalization
	sc = StandardScaler()
	sc.fit(x_value)
	x_norm = sc.transform(x_value)

	# No array
	NoArr = ['0']
	# AD array
	ADArr = ['1']

	x_train_data_List = []
	y_train_data_List = []
	x_test_data_List = []
	y_test_data_List = []

	print("xy_me_ge_values size => row: ", len(xy_me_ge_values))
	print("xy_me_ge_values size [0] => col: ", len(xy_me_ge_values[0]))

	rowSize = len(xy_me_ge_values)
	colSize = len(xy_me_ge_values[0])
	trainSize = int(rowSize * (train_test_ratio))
	testSize = rowSize - trainSize
	print("trainSize: ", trainSize.__str__(), "\t" + "testSize: ", testSize.__str__()) ## 1600 vs 400

	trNoMax = trainSize/2
	trADMax = trainSize/2
	teNoMax = testSize/2
	teADMax = testSize/2

	trNoCnt = 0
	trADCnt = 0
	teNoCnt = 0
	teADCnt = 0

	idx_pool = []
	for i in range(0, rowSize):
		idx_pool.append(i)

	#print(idx_pool)
	rd.shuffle(idx_pool)

	while idx_pool:
		i = idx_pool.pop()
		label = xy_me_ge_values[i][colSize - 1:colSize]

		# Normal - train
		if(np.array_equal(label, NoArr) and trNoCnt < trNoMax):
			x_tmpRow = x_norm[i]
			#y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
			x_train_data_List.append(x_tmpRow)
			y_train_data_List.append([1, 0])
			trNoCnt += 1

		# AD - train
		if (np.array_equal(label, ADArr) and trADCnt < trADMax):
			x_tmpRow = x_norm[i]
			#y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
			x_train_data_List.append(x_tmpRow)
			y_train_data_List.append([0, 1])
			trADCnt += 1

		# Normal - test
		if(np.array_equal(label, NoArr) and teNoCnt < teNoMax):
			x_tmpRow = x_norm[i]
			#y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
			x_test_data_List.append(x_tmpRow)
			y_test_data_List.append([1, 0])
			teNoCnt += 1

		# AD - test
		if (np.array_equal(label, ADArr) and teADCnt < teADMax):
			x_tmpRow = x_norm[i]
			#y_tmpRow = xy_me_ge_values[i, colSize-1:colSize]
			x_test_data_List.append(x_tmpRow)
			y_test_data_List.append([0, 1])
			teADCnt += 1


	np.delete(xy_me_ge_values, np.s_[:])

	sampleInfo = "trainSize_No: " + trNoCnt.__str__() + "\t" + " trainSize_AD: " + trADCnt.__str__() + " testSize_No: " + teNoCnt.__str__(), "\t" + " testSize_AD: " + teADCnt.__str__()
	print(sampleInfo)


	return np.array(x_train_data_List), np.array(y_train_data_List), np.array(x_test_data_List), np.array(y_test_data_List)


## neural network
def neural_network(num_hidden, size_layer, learning_rate, dropout_rate, batch_size=256):
	activation = 2
	def activate(activation, first_layer, second_layer, bias):
		if activation == 0:
			activation = tf.nn.sigmoid
			layer = activation(tf.matmul(first_layer, second_layer) + bias)

		elif activation == 1:
			activation = tf.nn.tanh
			layer = activation(tf.matmul(first_layer, second_layer) + bias)

		else:
			activation = tf.nn.relu
			layer = activation(tf.matmul(first_layer, second_layer) + bias)

		return tf.nn.dropout(layer, dropout_rate)

	tf.reset_default_graph()

	# define placeholder for X, Y
	X = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])  # 191 features
	Y = tf.placeholder(tf.int32, shape=[None, 1])  # 2 class [0 or 1] in one label

	classes = 2
	Y_one_hot = tf.one_hot(Y, classes)
	Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])

	## initialization of each layers
	input_layer = tf.get_variable("w_1", shape=[x_data.shape[1], size_layer], initializer=xavier_init(x_data.shape[1], size_layer)) ## initializer=xavier_init ## for L1
	biased_layer = tf.Variable(tf.random_normal([size_layer])) ## for L1
	output_layer = tf.get_variable("w_o", shape=[size_layer, onehot.shape[1]], initializer=xavier_init(size_layer, onehot.shape[1]))  ## initializer=xavier_init ## for L1
	biased_output = tf.Variable(tf.random_normal([onehot.shape[1]]))
	layers, biased = [], []

	## build layers
	for i in range(num_hidden - 1):
		layers.append(tf.get_variable(name="w_" + str(i+2), shape=[size_layer, size_layer], initializer=xavier_init(size_layer, size_layer))) ## initializer=xavier_init
		biased.append(tf.Variable(tf.random_normal([size_layer])))

	first_l = activate(activation, X, input_layer, biased_layer) ## for L1
	next_l = activate(activation, first_l, layers[0], biased[0]) ## for L1

	for i in range(1, num_hidden - 1):
		next_l = activate(activation, next_l, layers[i], biased[i])

	last_l = tf.add(tf.matmul(next_l, output_layer), biased_output)

	## define loss function, optimization, accuracy
	hypothesis = tf.nn.softmax(last_l)
	cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=last_l, labels=Y_one_hot) # original
	cost = tf.reduce_mean(cost_i)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	prediction = tf.argmax(hypothesis, 1)
	correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	## start to run TF
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	trainingEpoch = 400
	for i in range(trainingEpoch):
		## training with entire dataset
		"""
		sess.run(optimizer, feed_dict={X: x_train, Y: y_train})
		loss, acc = sess.run([cost, accuracy], feed_dict={X: x_train, Y: y_train})
		"""

		## training with batch dataset
		acc = 0
		loss = 0
		for n in range(0, (x_train.shape[0] // batch_size) * batch_size, batch_size):
			_, tr_bat_loss = sess.run([optimizer, cost], feed_dict={X: x_train[n: n + batch_size, :], Y: y_train[n: n + batch_size, :]})
			acc += sess.run(accuracy, feed_dict={X: x_train[n: n + batch_size, :], Y: y_train[n: n + batch_size, :]})
			loss += tr_bat_loss
		loss /= (x_train.shape[0] // batch_size)
		acc /= (x_train.shape[0] // batch_size)

		## display results
		print("epoch: " + str(i) + "\t" + "train_loss: " + str(loss) + "\t" + "train_acc: " + str(acc) + "\t" + "test_acc: " + str(sess.run(accuracy, feed_dict={X: x_test, Y: y_test})))

	## final results
	TEST_COST = sess.run(cost, feed_dict={X: x_test, Y: y_test})
	TEST_ACC = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
	COST = loss
	ACC = acc

	return COST, TEST_COST, ACC, TEST_ACC


## generate neural network
def generate_nn(num_hidden, size_layer, learning_rate, dropout_rate):
	global accbest
	param = {
		'num_hidden' : int(np.around(num_hidden)),
		'size_layer' : int(np.around(size_layer)),
		'learning_rate' : max(min(learning_rate, 1), 0.0001),
		'dropout_rate' : max(min(dropout_rate, 0.99), 0),
		#'beta' : max(min(beta, 0.5), 0.01),
		#'activation': int(np.around(activation))
	}
	print("\nSearch parameters %s" % (param))
	learning_cost, valid_cost, learning_acc, valid_acc = neural_network(**param)
	print("stop after 200 iteration with train cost %f, valid cost %f, train acc %f, valid acc %f" % (learning_cost, valid_cost, learning_acc, valid_acc))
	print("Search parameters %s\t stop after 400 iteration with train cost\t%f\t test cost\t%f\t train acc\t%f\t test acc\t%f" % (param, learning_cost, valid_cost, learning_acc, valid_acc), file=log_file)
	log_file.flush()

	if (valid_acc > accbest):
		costbest = valid_acc
	return valid_acc


########################################################################################################################
## start Bayesian optimization
Thres_lfc = 1
Thres_pval = 0.01
degSet = getDEG_limma("../../dataset/DEG_list.tsv", Thres_lfc, Thres_pval)
dmgSet, cpgSet = getDMG("../../dataset/DMP_list.tsv") ## DMG: only LFC 1, Pval 0.01
its_geneSet = degSet & dmgSet

## extract genes, methylation positions
XY_gxpr = applyDimReduction_DEG_intersectGene("../../dataset/allforDNN_ge.txt", its_geneSet, "../../dataset/DEG_list.tsv", Thres_lfc, Thres_pval)
XY_meth = applyDimReduction_DMP_intersectGene("../../dataset/allforDNN_me.txt", its_geneSet, "../../dataset/DMP_list.tsv")

input_data_mode = "all"
if input_data_mode == "all":
	## integrate two heterogenous dataset
	XY_gxpr_meth = buildIntegratedDataset_DNN(XY_gxpr, XY_meth, "unbalanced")
else:
	## integrate two heterogenous dataset with randomly selected N samples
	nSamples = 2000 ## for each label
	XY_gxpr_meth = buildIntegratedDataset_DNN_selectN(XY_gxpr, XY_meth, "unbalanced", nSamples)

print("final XY_gxpr_meth: " + str(XY_gxpr_meth.shape)) ## row: 2000 samples col: 193 (191 features + 2 labels)
print(XY_gxpr_meth)

## ratio of train & test

kf = KFold(n_splits=5)
k = 1
X = XY_gxpr_meth[:, 1:-1]
Y = XY_gxpr_meth[:, -1:]

print("X: " + str(X.shape))
print(X)
print("Y: " + str(Y.shape))
print(Y)

for tr_idx, te_idx in kf.split(XY_gxpr_meth):
	print("\nk: " + str(k))

	x_train, x_test = X[tr_idx], X[te_idx]
	y_train, y_test = Y[tr_idx], Y[te_idx]

	y_train = changeEncoding(y_train)
	y_test = changeEncoding(y_test)

	print("x_train: " + str(x_train.shape))
	print("y_train: " + str(y_train.shape))
	print("x_test: " + str(x_test.shape))
	print("y_test: " + str(y_test.shape))

	x_train = x_train.astype(np.float)
	x_test = x_test.astype(np.float)
	y_train = y_train.astype(np.int)
	y_test = y_test.astype(np.int)


	## entire X and Y dataset
	x_data = np.concatenate((x_train, x_test), axis=0)
	onehot = np.concatenate((y_train, y_test), axis=0)

	xy_train = np.concatenate((x_train, y_train), axis=1)
	xy_test = np.concatenate((x_test, y_test), axis=1)

	print("\n")
	print("x_data: " + str(x_data.shape))
	print("onehot: " + str(onehot.shape))
	print("xy_train: " + str(xy_train.shape))
	print("xy_test: " + str(xy_test.shape))
	print("\n\n\n")

	## test basic machine learning model
	#doMachineLearning_single_Kfold(xy_train, xy_test, "./dataset/BO_input_ML_test_result.txt", 1)

	## file name for final result
	log_filename = '../../results/k_fold_train_test_results/nn-bayesian_hpSearch_' + str(k) + '.log'
	if os.path.exists(log_filename):
		os.remove(log_filename)

	log_file = open(log_filename, 'a')
	accbest = 0.0

	## divide data into X and Y
	x_train, y_train = partitionTrainTest_ML_for_CV_DNN(xy_train)
	x_test, y_test = partitionTrainTest_ML_for_CV_DNN(xy_test)

	## do BayesianOptimization
	NN_BAYESIAN = BayesianOptimization(generate_nn,
									   {'num_hidden': (7, 11),
										'size_layer': (250, 350),
										'learning_rate': (0.01, 0.2),
										'dropout_rate': (0.6, 0.9),
										#'beta': (0.01, 0.49),
										#'activation': (2, 2)
										})
	NN_BAYESIAN.maximize(init_points = 30, n_iter = 50, acq = 'ei', xi = 0.0)

	print("\n\n\n")
	print('Max: ' + str(NN_BAYESIAN.max))
	print('Res: ' + str(NN_BAYESIAN.res))

	k += 1

