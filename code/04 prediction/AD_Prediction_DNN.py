####################################################################################################################################################
# AD_Prediction_by_DNN.py
# Author: Chihyun Park
# Email: chihyun.park@yonsei.ac.kr
# Date created: 11/02/2018
# Date lastly modified: 7/1/2019
# Purpose: To construct AD prediction model using DNN
# input
# 		for each K fold dataset
#       1. gene expression (samples x genes) with label (AD, Normal)
#       2. DNA methylation (samples x CpG probes) with label (AD, Normal)
#       3. DEG list (Normal vs AD)
#       4. DMP list (Normal vs AD)
# output
#       prediction performance by DNN and various machine learning algorithms while varying dimension reduction (feature selection) algorithms
####################################################################################################################################################

import tensorflow as tf
import numpy as np
import os.path
import argparse
import itertools
import csv
import pandas as pd

import sys
import os
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif


## initialization of DNN
def xavier_init(n_inputs, n_outputs, uniform=True):
	if uniform:
		init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
		return tf.random_uniform_initializer(-init_range, init_range)
	else:
		stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
	return tf.truncated_normal_initializer(stddev=stddev)


## integrate multiomics datasets
def buildIntegratedDataset_DNN(xy_gxpr, xy_meth, mode):
	print("buildIntegratedDataset")
	xy_gxpr_meth = []

	n_row_g, n_col_g = xy_gxpr.shape
	n_row_m, n_col_m = xy_meth.shape

	# build random index pair set
	idxSet_No = set()
	idxSet_AD = set()

	NoArr = [1., 0.]
	ADArr = [0., 1.]
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


	print("xy_gxpr_meth: " + len(xy_gxpr_meth).__str__())

	"""
	for idx in range(0, 10):
		xy = xy_gxpr_meth[idx]
		geneSet_str = ";".join(str(x) for x in xy)
		print(geneSet_str)
	"""
	xy_me_ge_values = np.array(xy_gxpr_meth)
	print(xy_me_ge_values.shape)

	return xy_me_ge_values


## load DEG by limma
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


## do feature selection with DEG
def applyFeatSel_DEG_intersectGene(infilename, geneSet):
	selected_genelist = ['SampleID']

	for gene in geneSet:
		selected_genelist.append(gene)

	# Label_No	Label_AD
	selected_genelist.append('Label_No')
	selected_genelist.append('Label_AD')

	xy_all_df = pd.read_csv(infilename, sep='\t', usecols=selected_genelist)
	xy = xy_all_df.as_matrix()

	xy_values = xy[:, 1:-2]
	xy_labels = xy[:, -2:]

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

	X_embedded = xy_values
	XY_embedded = np.append(X_embedded, xy_labels, axis=1)

	#print(XY_embedded)
	print(XY_embedded.shape)

	return XY_embedded



## do feature selection with DMP
def applyFeatSel_DMP_intersectGene(infilename, geneSet, geneCpgSet_map):
	selected_cpglist = ['SampleID']

	for gene, cpgSet in geneCpgSet_map.items():
		if gene in geneSet:
			for cpg in cpgSet:
				selected_cpglist.append(cpg)

	selected_cpglist.append('Label_No')
	selected_cpglist.append('Label_AD')

	xy_all_df = pd.read_csv(infilename, sep='\t', usecols=selected_cpglist)
	xy = xy_all_df.as_matrix() ## sampleID, expr + label 2 columns

	xy_tp = np.transpose(xy)
	xy_values = xy[:, 1:-2]
	xy_labels = xy[:, -2:]

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

	X_embedded = xy_values
	XY_embedded = np.append(X_embedded, xy_labels, axis=1)

	#print(XY_embedded.shape)
	#print(XY_embedded)

	return XY_embedded


## do feature selection with PCA
def applyDimReduction_PCA(infilename, num_comp, scatterPlot_fn, mode):
	print("applyDimReduction PCA")
	xy = np.genfromtxt(infilename, unpack=True, delimiter='\t', dtype=str)
	xy_tp = np.transpose(xy)
	print("xy_tp: " + str(xy_tp.shape))

	xy_featureList = xy_tp[0, 1:]
	#print(xy_featureList)

	xy_values = xy_tp[1:, 1:-2]
	xy_labels = xy_tp[1:, -2:]

	xy_values = xy_values.astype(np.float)
	xy_labels = xy_labels.astype(np.float)

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

	#print("xy_tp row: " + len(xy_tp).__str__() + "\t" + " col: " + len(xy_tp[0]).__str__())
	#print("xy_values row: " + len(xy_values).__str__() + "\t" + " col: " + len(xy_values[0]).__str__())
	#print("xy_labels row: " + len(xy_labels).__str__() + "\t" + " col: " + len(xy_labels[0]).__str__())

	print("xy_values: " + str(xy_values.shape))
	print("xy_labels: " + str(xy_labels.shape))

	# print(xy_values)
	# print(xy_labels_1_column)
	# apply PCA

	if num_comp >= xy_values.shape[0]:
		num_comp = xy_values.shape[0]

	pca = PCA(n_components=num_comp, svd_solver='full')
	pca.fit(xy_values)
	X_embedded = pca.transform(xy_values)
	print(X_embedded.shape)
	XY_embedded = np.append(X_embedded, xy_labels, axis=1)
	print(XY_embedded.shape)

	return XY_embedded, num_comp


## do feature selection with t-SNE
def applyDimReduction_TSNE(infilename, num_comp, scatterPlot_fn, mode):
	print("applyDimReduction t-SNE")
	xy = np.genfromtxt(infilename, unpack=True, delimiter='\t', dtype=str)
	xy_tp = np.transpose(xy)
	print("xy_tp: " + str(xy_tp.shape))
	xy_featureList = xy_tp[0, 1:]
	#print(xy_featureList)

	xy_values = xy_tp[1:, 1:-2]
	xy_labels = xy_tp[1:, -2:]

	xy_values = xy_values.astype(np.float)
	xy_labels = xy_labels.astype(np.float)

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

	#print("xy_tp row: " + len(xy_tp).__str__() + "\t" + " col: " + len(xy_tp[0]).__str__())
	#print("xy_values row: " + len(xy_values).__str__() + "\t" + " col: " + len(xy_values[0]).__str__())
	#print("xy_labels row: " + len(xy_labels).__str__() + "\t" + " col: " + len(xy_labels[0]).__str__())

	print("xy_values: " + str(xy_values.shape))
	print("xy_labels: " + str(xy_labels.shape))

	X_embedded = TSNE(n_components=num_comp, method='exact').fit_transform(xy_values)
	XY_embedded = np.append(X_embedded, xy_labels, axis=1)
	print("XY_embedded: " + XY_embedded.shape.__str__())

	return XY_embedded


## split input data into x and y
def partitionTrainTest_ML_for_CV_DNN(xy_me_ge_values):
	np.random.shuffle(xy_me_ge_values)

	x_data_List = []
	y_data_List = []

	colSize = len(xy_me_ge_values[0])

	for i in range(len(xy_me_ge_values)):
		x_tmpRow = xy_me_ge_values[i, 1:colSize - 1]
		y_tmpRow = xy_me_ge_values[i, colSize - 1:colSize]

		x_data_List.append(x_tmpRow)
		y_data_List.append(y_tmpRow)

	return np.array(x_data_List), np.array(y_data_List)


## construct DNN with 8 hidden layers and training/test
def doDNN_8(xy_train, xy_test, outfilename, total_epoch, determine_variable_reuse):
	print("doDNN_8")
	size_layer = 306
	learning_rate = 0.03
	drop_out_rate = 0.86

	fout = open(outfilename, 'w')
	fout.write("Do DNN with DNA methylation and Gene expression\n")

	print("xy_train: " + str(xy_train.shape))
	#print(xy_train[0,])
	print("xy_test" + str(xy_test.shape))
	#print(xy_test[0,])

	x_train_data, y_train_data = partitionTrainTest_ML_for_CV_DNN(xy_train)
	x_test_data, y_test_data = partitionTrainTest_ML_for_CV_DNN(xy_test)

	x_train_data = x_train_data.astype(np.float32)
	x_test_data = x_test_data.astype(np.float32)
	y_train_data = y_train_data.astype(np.int)
	y_test_data = y_test_data.astype(np.int)

	colSize = x_train_data.shape[1]
	print("colSize: " + str(colSize))
	NoArr = [0]
	ADArr = [1]
	tr_no_cnt = 0
	tr_ad_cnt = 0
	te_no_cnt = 0
	te_ad_cnt = 0

	for label in y_train_data:
		if np.array_equal(label, NoArr):
			tr_no_cnt += 1
		if np.array_equal(label, ADArr):
			tr_ad_cnt += 1

	for label in y_test_data:
		if np.array_equal(label, NoArr):
			te_no_cnt += 1
		if np.array_equal(label, ADArr):
			te_ad_cnt += 1

	trainSize = tr_no_cnt + tr_ad_cnt

	nSize_L1 = size_layer
	nSize_L2 = size_layer
	nSize_L3 = size_layer
	nSize_L4 = size_layer
	nSize_L5 = size_layer
	nSize_L6 = size_layer
	nSize_L7 = size_layer
	nSize_L8 = size_layer
	nSize_L9 = 2

	###################################
	print("x,y train")
	print(x_train_data.shape, y_train_data.shape)
	print("x,y test")
	print(x_test_data.shape, y_test_data.shape)
	classes = 2 # number of class label : 0 ~ 1

	###################################
	if ("yes" in determine_variable_reuse):
		with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):
			print("reuse true")

			# define placeholder for X, Y
			X = tf.placeholder(tf.float32, shape=[None, x_train_data.shape[1]])  # 191 features
			Y = tf.placeholder(tf.int32, shape=[None, 1])  # 2 class [0 or 1] in one label

			classes = 2
			Y_one_hot = tf.one_hot(Y, classes)
			Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])

			"""
			X = tf.placeholder(tf.float32, shape=[None, colSize])  # 19448 features
			Y = tf.placeholder(tf.int32, shape=[None, 1])  # 2 labels [0 or 1]
			Y_one_hot = tf.one_hot(Y, classes)  # one-hot : rank가 1 증가됨
			print('one-hot :', Y_one_hot)
			Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])  # reshape로 shape 변환
			print('reshape :', Y_one_hot)
			"""

			W1 = tf.get_variable("W1_ge", shape=[colSize, nSize_L1], initializer=xavier_init(colSize, nSize_L1))
			W2 = tf.get_variable("W2_ge", shape=[nSize_L1, nSize_L2], initializer=xavier_init(nSize_L1, nSize_L2))
			W3 = tf.get_variable("W3_ge", shape=[nSize_L2, nSize_L3], initializer=xavier_init(nSize_L2, nSize_L3))
			W4 = tf.get_variable("W4_ge", shape=[nSize_L3, nSize_L4], initializer=xavier_init(nSize_L3, nSize_L4))
			W5 = tf.get_variable("W5_ge", shape=[nSize_L4, nSize_L5], initializer=xavier_init(nSize_L4, nSize_L5))
			W6 = tf.get_variable("W6_ge", shape=[nSize_L5, nSize_L6], initializer=xavier_init(nSize_L5, nSize_L6))
			W7 = tf.get_variable("W7_ge", shape=[nSize_L6, nSize_L7], initializer=xavier_init(nSize_L6, nSize_L7))
			W8 = tf.get_variable("W8_ge", shape=[nSize_L7, nSize_L8], initializer=xavier_init(nSize_L7, nSize_L8))
			W9 = tf.get_variable("W9_ge", shape=[nSize_L8, nSize_L9], initializer=xavier_init(nSize_L8, nSize_L9))

			b1 = tf.Variable(tf.random_normal([nSize_L1]), name="Bias1")
			b2 = tf.Variable(tf.random_normal([nSize_L2]), name="Bias2")
			b3 = tf.Variable(tf.random_normal([nSize_L3]), name="Bias3")
			b4 = tf.Variable(tf.random_normal([nSize_L4]), name="Bias4")
			b5 = tf.Variable(tf.random_normal([nSize_L5]), name="Bias5")
			b6 = tf.Variable(tf.random_normal([nSize_L6]), name="Bias6")
			b7 = tf.Variable(tf.random_normal([nSize_L7]), name="Bias7")
			b8 = tf.Variable(tf.random_normal([nSize_L8]), name="Bias8")
			b9 = tf.Variable(tf.random_normal([nSize_L9]), name="Bias9")

			dropout_rate = tf.placeholder("float")  # sigmoid or relu


	if ("no" in determine_variable_reuse):
		print("first use")
		tf.reset_default_graph()

		# define placeholder for X, Y
		X = tf.placeholder(tf.float32, shape=[None, x_train_data.shape[1]])  # 191 features
		Y = tf.placeholder(tf.int32, shape=[None, 1])  # 2 class [0 or 1] in one label

		classes = 2
		Y_one_hot = tf.one_hot(Y, classes)
		Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])

		"""
		X = tf.placeholder(tf.float32, shape=[None, colSize])  # 19448 features
		Y = tf.placeholder(tf.int32, shape=[None, 1])  # 2 labels [0 or 1]
		Y_one_hot = tf.one_hot(Y, classes)  # one-hot : rank가 1 증가됨
		print('one-hot :', Y_one_hot)
		Y_one_hot = tf.reshape(Y_one_hot, [-1, classes])  # reshape로 shape 변환
		print('reshape :', Y_one_hot)
		"""

		W1 = tf.get_variable("W1_ge", shape=[colSize, nSize_L1], initializer=xavier_init(colSize, nSize_L1))
		W2 = tf.get_variable("W2_ge", shape=[nSize_L1, nSize_L2], initializer=xavier_init(nSize_L1, nSize_L2))
		W3 = tf.get_variable("W3_ge", shape=[nSize_L2, nSize_L3], initializer=xavier_init(nSize_L2, nSize_L3))
		W4 = tf.get_variable("W4_ge", shape=[nSize_L3, nSize_L4], initializer=xavier_init(nSize_L3, nSize_L4))
		W5 = tf.get_variable("W5_ge", shape=[nSize_L4, nSize_L5], initializer=xavier_init(nSize_L4, nSize_L5))
		W6 = tf.get_variable("W6_ge", shape=[nSize_L5, nSize_L6], initializer=xavier_init(nSize_L5, nSize_L6))
		W7 = tf.get_variable("W7_ge", shape=[nSize_L6, nSize_L7], initializer=xavier_init(nSize_L6, nSize_L7))
		W8 = tf.get_variable("W8_ge", shape=[nSize_L7, nSize_L8], initializer=xavier_init(nSize_L7, nSize_L8))
		W9 = tf.get_variable("W9_ge", shape=[nSize_L8, nSize_L9], initializer=xavier_init(nSize_L8, nSize_L9))

		b1 = tf.Variable(tf.random_normal([nSize_L1]), name="Bias1")
		b2 = tf.Variable(tf.random_normal([nSize_L2]), name="Bias2")
		b3 = tf.Variable(tf.random_normal([nSize_L3]), name="Bias3")
		b4 = tf.Variable(tf.random_normal([nSize_L4]), name="Bias4")
		b5 = tf.Variable(tf.random_normal([nSize_L5]), name="Bias5")
		b6 = tf.Variable(tf.random_normal([nSize_L6]), name="Bias6")
		b7 = tf.Variable(tf.random_normal([nSize_L7]), name="Bias7")
		b8 = tf.Variable(tf.random_normal([nSize_L8]), name="Bias8")
		b9 = tf.Variable(tf.random_normal([nSize_L9]), name="Bias9")

		dropout_rate = tf.placeholder("float")  # sigmoid or relu

	Layer1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(X, W1), b1)), dropout_rate)
	Layer2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer1, W2), b2)), dropout_rate)
	Layer3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer2, W3), b3)), dropout_rate)
	Layer4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer3, W4), b4)), dropout_rate)
	Layer5 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer4, W5), b5)), dropout_rate)
	Layer6 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer5, W6), b6)), dropout_rate)
	Layer7 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer6, W7), b7)), dropout_rate)
	Layer8 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(Layer7, W8), b8)), dropout_rate)
	Layer9 = tf.add(tf.matmul(Layer8, W9), b9)

	# define logit, hypothesis
	logits = Layer9
	print("logits: " , logits)
	#sys.exit()

	hypothesis = tf.nn.softmax(logits)

	# cross entropy cost function
	cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
	cost = tf.reduce_mean(cost_i)
	##########################################################


	# Gradient Descent optimizer
	train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	prediction = tf.argmax(hypothesis, 1)
	correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	training_epochs = total_epoch #13020  # 12000 # 4000

	##############################################
	print("DNN structure")
	print("Relu / Softmax / Onehot vector encoding")
	print("nSize_L1: " + nSize_L1.__str__())
	print("nSize_L2: " + nSize_L2.__str__())
	print("nSize_L3: " + nSize_L3.__str__())
	print("nSize_L4: " + nSize_L4.__str__())
	print("nSize_L5: " + nSize_L5.__str__())
	print("nSize_L6: " + nSize_L6.__str__())
	print("nSize_L7: " + nSize_L7.__str__())
	print("nSize_L8: " + nSize_L8.__str__())
	print("nSize_L9: " + nSize_L9.__str__())

	print("DNN parameters")
	print("learning_rate: {:1.5f}".format(learning_rate))
	print("training_epochs: " + training_epochs.__str__())
	print("dr_rate: " + drop_out_rate.__str__())
	##############################################

	# fout all information
	fout.write("DNN structure\n")
	fout.write("Relu / Softmax / Onehot vector encoding\n")
	fout.write("trainSize: " + trainSize.__str__() + "\n")
	fout.write("nSize_L1: " + nSize_L1.__str__() + "\n")
	fout.write("nSize_L2: " + nSize_L2.__str__() + "\n")
	fout.write("nSize_L3: " + nSize_L3.__str__() + "\n")
	fout.write("nSize_L4: " + nSize_L4.__str__() + "\n")
	fout.write("nSize_L5: " + nSize_L5.__str__() + "\n")
	fout.write("nSize_L6: " + nSize_L6.__str__() + "\n")
	fout.write("nSize_L7: " + nSize_L7.__str__() + "\n")
	fout.write("nSize_L8: " + nSize_L8.__str__() + "\n")
	fout.write("nSize_L9: " + nSize_L9.__str__() + "\n")
	fout.write("DNN parameters" + "\n")
	fout.write("learning_rate: {:1.5f}".format(learning_rate) + "\n")
	fout.write("training_epochs: " + training_epochs.__str__() + "\n")
	fout.write("dr_rate: " + drop_out_rate.__str__() + "\n")
	##############################################

	test_accuracy_list = []
	test_loss_list = []
	train_accuracy_list = []

	# Launch the graph
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	early_stop_epoch_test_acc_map = {}
	early_stop_epoch_test_loss_map = {}
	early_stop_epoch_test_auroc_map = {}
	stop_epoch = 0

	for epoch in range(training_epochs):
		sess.run(train, feed_dict={X: x_train_data, Y: y_train_data, dropout_rate: drop_out_rate})

		# if epoch % 10 == 0 or epoch == training_epochs-1:
		loss, acc = sess.run([cost, accuracy], feed_dict={X: x_train_data, Y: y_train_data, dropout_rate: drop_out_rate})
		#print("epoch: {:5}\tLoss: {:.3f}\tTraining Acc: {:.2%}".format(epoch, loss, acc))
		#trResult = "epoch:\t" + epoch.__str__() + "\t" + "cost:\t" + loss.__str__() + "\t" + "Training accuracy:\t" + acc.__str__()
		#fout.write(trResult + "\n")
		test_loss, test_acc = sess.run([cost, accuracy], feed_dict={X: x_test_data, Y: y_test_data, dropout_rate: 1.0})

		if epoch % 10 == 0 or epoch == training_epochs - 1:
			print("epoch: " + str(epoch) + "\t" + "train_loss: " + str(loss) + "\t" + "train_acc: " + str(acc) + "\t" + "test_acc: " + str(test_acc))

		fout.write("epoch:\t" + str(epoch) + "\t" + "train_loss:\t" + str(loss) + "\t" + "train_acc:\t" + str(acc) + "\t" + "test_acc:\t" + str(test_acc) + "\n")

		test_accuracy_list.append(test_acc)
		train_accuracy_list.append(acc)
		test_loss_list.append(test_loss)

		last_n_elements = 10
		if epoch > 100:
			# print(accuracy_list[int(round(len(accuracy_list) / 5)):])
			#test_acc_mean = np.mean(accuracy_list[int(round(len(accuracy_list) / 3)):]
			test_acc_mean = np.mean(test_accuracy_list[-last_n_elements:])
			train_acc_mean = np.mean(train_accuracy_list[-last_n_elements:])

			if (test_acc - test_acc_mean) <= 0.001 and (acc - train_acc_mean) >= 0.01:
				es_test_acc = test_accuracy_list[-last_n_elements]
				es_epoch = epoch - last_n_elements
				es_test_loss = test_loss_list[-last_n_elements]

				print("early stopping test_acc: " + str(es_test_acc))
				print("early stopping @ " + str(es_epoch))

				pred = sess.run(prediction, feed_dict={X: x_test_data, dropout_rate: 1.0})
				pred_arr = np.asarray(pred)
				test_auroc = roc_auc_score(y_test_data, pred_arr)

				early_stop_epoch_test_acc_map[es_epoch] = es_test_acc
				early_stop_epoch_test_loss_map[es_epoch] = es_test_loss
				early_stop_epoch_test_auroc_map[es_epoch] = test_auroc
				stop_epoch = es_epoch
				break

	print("@@ Optimization Finished: sess.run with test data @@")
	pred = sess.run(prediction, feed_dict={X: x_test_data, dropout_rate: 1.0})

	no_cnt_true = 0
	ad_cnt_true = 0
	for y in y_test_data:
		if y == 0:
			no_cnt_true += 1
		else:
			ad_cnt_true += 1

	# AD: positive conditon, No: negative condition
	tp_cnt = 0
	fp_cnt = 0
	fn_cnt = 0
	tn_cnt = 0
	#print(pred)
	#print("pred len: " + len(pred).__str__())

	for p, y in zip(pred, y_test_data.flatten()):
		# print(p.__str__() + "\t" + y.__str__())
		if p == int(y):
			if (y == 1):
				tp_cnt += 1
			else:
				tn_cnt += 1

		if p != int(y):
			if (y == 1):
				fn_cnt += 1
			else:
				fp_cnt += 1

	print("AD cnt: " + ad_cnt_true.__str__() + "\t" + "No cnt: " + no_cnt_true.__str__())
	print("TP: " + tp_cnt.__str__() + "\t" + "FP: " + fp_cnt.__str__())
	print("FN: " + fn_cnt.__str__() + "\t" + "TN: " + tn_cnt.__str__())
	# print(sess.run(prediction, feed_dict={X: x_test_data, dropout_rate: 1.0}))
	test_accuracy_by_tptnfpfn = float((tp_cnt + tn_cnt)/(tp_cnt + fp_cnt + fn_cnt + tn_cnt))
	print("test_accuracy_by_tptnfpfn: " + str(test_accuracy_by_tptnfpfn))

	if stop_epoch != 0:
		eStop_acc = early_stop_epoch_test_acc_map[stop_epoch]
		eStop_loss = early_stop_epoch_test_loss_map[stop_epoch]
		eStop_auroc = early_stop_epoch_test_auroc_map[stop_epoch]

		print("eStop_best_epoch: " + str(stop_epoch))
		print("eStop_best_acc: " + str(eStop_acc))

		print("@@ 1 Optimization Finished with accuracy.eval @@")
		test_loss, test_acc = sess.run([cost, accuracy], feed_dict={X: x_test_data, Y: y_test_data, dropout_rate: 1.0})

		print("Test Loss:\t" + str(test_loss) + "\n" +
			  "Test Acc:\t" + str(test_acc) + "\n" +
			  "Early Stop Epoch:\t" + str(stop_epoch) + "\n" +
			  "Early Stop Test loss:\t" + str(eStop_loss) + "\n" +
			  "Early Stop Test Acc:\t" + str(eStop_acc) + "\n" +
			  "Early Stop Test AUROC:\t" + str(eStop_auroc) + "\n")
		fout.write("Test Loss:\t" + str(test_loss) + "\n" +
				   "Test Acc:\t" + str(test_acc) + "\n" +
				   "Early Stop Epoch:\t" + str(stop_epoch) + "\n" +
				   "Early Stop Test loss:\t" + str(eStop_loss) + "\n" +
				   "Early Stop Test Acc:\t" + str(eStop_acc) + "\n" +
				   "Early Stop Test AUROC:\t" + str(eStop_auroc) + "\n")

	else:
		eStop_acc = test_accuracy_list[-1]
		eStop_loss = test_loss_list[-1]

		pred = sess.run(prediction, feed_dict={X: x_test_data, dropout_rate: 1.0})
		pred_arr = np.asarray(pred)
		eStop_auroc = roc_auc_score(y_test_data, pred_arr)

		print("@@ 1 Optimization Finished with accuracy.eval @@")
		test_loss, test_acc = sess.run([cost, accuracy], feed_dict={X: x_test_data, Y: y_test_data, dropout_rate: 1.0})
		print("Test Loss:\t" + str(test_loss) + "\n" +
			  "Test Acc:\t" + str(test_acc) + "\n" +
			  "Early Stop Epoch:\t" + str(stop_epoch) + "\n" +
			  "Early Stop Test loss:\t" + str(eStop_loss) + "\n" +
			  "Early Stop Test Acc:\t" + str(eStop_acc) + "\n" +
			  "Early Stop Test AUROC:\t" + str(eStop_auroc) + "\n")
		fout.write("Test Loss:\t" + str(test_loss) + "\n" +
				   "Test Acc:\t" + str(test_acc) + "\n" +
				   "Early Stop Epoch:\t" + str(stop_epoch) + "\n" +
				   "Early Stop Test loss:\t" + str(eStop_loss) + "\n" +
				   "Early Stop Test Acc:\t" + str(eStop_acc) + "\n" +
				   "Early Stop Test AUROC:\t" + str(eStop_auroc) + "\n")


	pred_arr = np.asarray(pred)
	print("@@ 2 ROC, AUC")
	print("y_test_data size: " + len(y_test_data).__str__() + "\t" + "pred_arr size: " + len(pred_arr).__str__())
	auc, update_op = tf.metrics.auc(y_test_data, pred_arr)
	print("AUC: " + auc.__str__())
	print("update_op: " + update_op.__str__())

	# by using sklearn
	auc_by_sk = roc_auc_score(y_test_data, pred_arr)
	print("auc_by_sk: " + auc_by_sk.__str__())
	print("precision: " + precision_score(y_test_data, pred_arr).__str__())
	print("recall: " + recall_score(y_test_data, pred_arr).__str__())
	print("f1_score: " + f1_score(y_test_data, pred_arr).__str__())
	print(confusion_matrix(y_test_data, pred_arr).__str__())
	fpr, tpr, thresholds = roc_curve(y_test_data, pred_arr)
	print("FPR")
	print(fpr.__str__())
	print("TPR")
	print(tpr.__str__())
	print("Thresholds")
	print(thresholds.__str__())

	sess.close()
	fout.close()


## get table between probe and gene
def getProbeGeneMap(mapTableFile):
	selected_cols = ["ID", "UCSC_RefGene_Name", "UCSC_RefGene_Group"]
	gpl_df = pd.read_csv(mapTableFile, sep='\t', skiprows=37, usecols=selected_cols)
	gpl_df["UCSC_RefGene_Name"].astype(str)
	gpl_df["UCSC_RefGene_Group"].astype(str)

	## remove NaN UCSC_RefGene_Name
	gpl_df = gpl_df.dropna(subset=['UCSC_RefGene_Name', 'UCSC_RefGene_Group'])

	## select interesting CpG
	interesting_TSS_list = ['TSS200', 'TSS1500']
	gpl_df["gene_symbol"] = gpl_df["UCSC_RefGene_Name"].apply(lambda x: x.split(';')[0] if ";" in x else '-')
	gpl_df["TSS"] = gpl_df["UCSC_RefGene_Group"].apply(lambda x: "TSS<2000" if interesting_TSS_list[0] in x or interesting_TSS_list[1] in x else '-')
	gpl_df = gpl_df[['ID', 'gene_symbol', 'TSS']]
	gpl_df = gpl_df.loc[gpl_df['TSS'] != "-"]
	gpl_df = gpl_df.loc[gpl_df['gene_symbol'] != "-"]

	#print("gpl_df: " + str(gpl_df.shape))
	## make dict
	cpg_geneSymbol_dict = dict(zip(gpl_df['ID'], gpl_df['gene_symbol']))
	#print("cpg_geneSymbol_dict: " + str(len(cpg_geneSymbol_dict.keys())))

	return cpg_geneSymbol_dict


## load DMG by limma
def getDMG_limma(filename, lfc, pval, probeGene_map):
	geneSet = set()
	f = open(filename, 'r')
	inCSV = csv.reader(f, delimiter="\t")
	header = next(inCSV)  # for header

	for row in inCSV:
		probe = row[0]
		logFC = float(row[1])
		Pvalue = float(row[4])  ## adj p-val : row[5]

		if abs(logFC) >= lfc and Pvalue < pval:
			if probe in probeGene_map.keys():
				gene = probeGene_map[probe]
				geneSet.add(gene)

	print("[limma - DMG] Number of gene set: " + str(len(geneSet)))
	#print("geneSet: " + str(geneSet))

	return geneSet


## load DEG, DMG by limma
def load_DEG_DMG(filePath, lfc, pval, mode, mapTableFile):
	geneSet = set()
	geneCpgSet_map = {}

	if mode == "DEG":
		degSet = getDEG_limma(filePath, lfc, pval)
		geneSet = degSet

	if mode == "DMP":
		probeGene_map = getProbeGeneMap(mapTableFile)
		dmgSet = getDMG_limma(filePath, lfc, pval, probeGene_map)  ## should be changed
		geneSet = dmgSet

		for p, g in probeGene_map.items():
			if g in geneCpgSet_map.keys():
				pset = geneCpgSet_map[g]
				pset.add(p)
				geneCpgSet_map[g] = pset
			else:
				pset = set()
				pset.add(p)
				geneCpgSet_map[g] = pset

	return geneSet, geneCpgSet_map


## main
def main(args):
	print("Deep Neural Network approach")
	input_dir = args.input  ## ./results/k_fold_train_test
	output_dir = args.output  ## ./results/k_fold_train_test_results
	if not os.path.exists(output_dir): os.mkdir(output_dir)

	for j in range(0, 1):
		if j == 0:
			mode = "unbalanced"  # "unbalanced"
		else:
			mode = "balanced"  # "unbalanced"

		print("mode: " + mode)

		## for each k
		for k in range(1, 6):
			print("\n\nK: " + str(k))

			## make directories
			## table 1
			dirPath_table1_ge = output_dir + "/k_" + str(k) + "/table_1/genExpr"
			if not os.path.exists(dirPath_table1_ge): os.mkdir(dirPath_table1_ge)
			dirPath_table1_me = output_dir + "/k_" + str(k) + "/table_1/meth"
			if not os.path.exists(dirPath_table1_me): os.mkdir(dirPath_table1_me)

			## table 2
			dirPath_table2_geme = output_dir + "/k_" + str(k) + "/table_2/genExpr_meth"
			if not os.path.exists(dirPath_table2_geme): os.mkdir(dirPath_table2_geme)

			## table 3
			dirPath_table3_deg = output_dir + "/k_" + str(k) + "/table_3/DEG"
			if not os.path.exists(dirPath_table3_deg): os.mkdir(dirPath_table3_deg)
			dirPath_table3_dmg = output_dir + "/k_" + str(k) + "/table_3/DMG"
			if not os.path.exists(dirPath_table3_dmg): os.mkdir(dirPath_table3_dmg)
			dirPath_table3_deg_dmg = output_dir + "/k_" + str(k) + "/table_3/DEG_DMG"
			if not os.path.exists(dirPath_table3_deg_dmg): os.mkdir(dirPath_table3_deg_dmg)

			##  table 4
			dirPath_table4_deg_dmg = output_dir + "/k_" + str(k) + "/table_4/DEG_DMG"
			if not os.path.exists(dirPath_table4_deg_dmg): os.mkdir(dirPath_table4_deg_dmg)


			################################################################################################################
			## make training, test dataset by our feature selection approach
			thresh_lfc_ge = 1
			thresh_pval_ge = 0.01
			thresh_lfc_me = 0.58
			thresh_pval_me = 0.01
			mapTableFile = "../../dataset/GPL13534-11288.txt"

			## training
			## load DEG, DMG for
			degSet, _ = load_DEG_DMG(input_dir + "/DEG/[train " + str(k) + "] AD DEG.tsv", thresh_lfc_ge, thresh_pval_ge, "DEG", mapTableFile)
			dmgSet, geneCpgSet_map = load_DEG_DMG(input_dir + "/DMP/[train " + str(k) + "] AD DMP.tsv", thresh_lfc_me, thresh_pval_me, "DMP", mapTableFile)

			its_geneSet = degSet & dmgSet
			print("its_geneSet: " + str(len(its_geneSet)))
			print("train_xy_gxpr: " + str(train_xy_gxpr.shape))
			print("train_xy_meth: " + str(train_xy_meth.shape))

			## our feature selection approach
			train_xy_gxpr = applyFeatSel_DEG_intersectGene(input_dir + "/XY_gexp_train_" + str(k) + "_ML_input.tsv", its_geneSet)
			train_xy_meth = applyFeatSel_DMP_intersectGene(input_dir + "/XY_meth_train_" + str(k) + "_ML_input.tsv", its_geneSet, geneCpgSet_map)
			test_xy_gxpr = applyFeatSel_DEG_intersectGene(input_dir + "/XY_gexp_test_" + str(k) + "_ML_input.tsv", its_geneSet)
			test_xy_meth = applyFeatSel_DMP_intersectGene(input_dir + "/XY_meth_test_" + str(k) + "_ML_input.tsv", its_geneSet, geneCpgSet_map)
			train_xy_gxpr_meth = buildIntegratedDataset_DNN(train_xy_gxpr, train_xy_meth, mode)  # "unbalanced"
			test_xy_gxpr_meth = buildIntegratedDataset_DNN(test_xy_gxpr, test_xy_meth, mode)
			dnn_result_output = output_dir + "/[" + str(k) + "]["+ mode + "] DNN_deg_dmg.tsv"
			doDNN_8(train_xy_gxpr_meth, test_xy_gxpr_meth, dnn_result_output, 1500, "no")

			## PCA, t-SNE + DNN
			print("\n\nExperiment 1~2. PCA, t-SNE + ML")
			num_of_dim_gxpr = train_xy_gxpr.shape[1] - 2
			num_of_dim_meth = train_xy_meth.shape[1] - 2
			num_of_dim_gexp_meth = train_xy_gxpr_meth.shape[1] - 2
			print("\n")
			print("# feature (gene expression):" + str(train_xy_gxpr.shape[1] - 2))
			print("# feature (DNA methylation):" + str(train_xy_meth.shape[1] - 2))
			print("\n")

			print("num_of_dim_gxpr: " + str(num_of_dim_gxpr))
			print("num_of_dim_meth: " + str(num_of_dim_meth))
			print("num_of_dim_gexp_meth: " + str(num_of_dim_gexp_meth))

			## t-SNE based approach
			train_xy_gxpr_tsne = applyDimReduction_TSNE(input_dir + "/XY_gexp_train_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/tsne_scatter_plot_gxpr", "train")
			test_xy_gxpr_tsne = applyDimReduction_TSNE(input_dir + "/XY_gexp_test_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/tsne_scatter_plot_gxpr", "test")
			train_xy_meth_tsne = applyDimReduction_TSNE(input_dir + "/XY_meth_train_" + str(k) + "_ML_input.tsv", num_of_dim_meth, dirPath_table1_me + "/tsne_scatter_plot_meth", "train")
			test_xy_meth_tsne = applyDimReduction_TSNE(input_dir + "/XY_meth_test_" + str(k) + "_ML_input.tsv", num_of_dim_meth, dirPath_table1_me + "/tsne_scatter_plot_meth", "test")
			train_xy_gxpr_meth_tsne = buildIntegratedDataset_DNN(train_xy_gxpr_tsne, train_xy_meth_tsne, mode)  # "unbalanced"
			test_xy_gxpr_meth_tsne = buildIntegratedDataset_DNN(test_xy_gxpr_tsne, test_xy_meth_tsne, mode)
			dnn_result_output = output_dir + "/[" + str(k) + "]["+ mode + "] DNN_deg_dmg [t-SNE].tsv"
			doDNN_8(train_xy_gxpr_meth_tsne, test_xy_gxpr_meth_tsne, dnn_result_output, 1500, "no")

			## PCA based approach
			train_xy_gxpr_pca, _ = applyDimReduction_PCA(input_dir + "/XY_gexp_train_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/pca_scatter_plot_gxpr", "train")
			test_xy_gxpr_pca, _ = applyDimReduction_PCA(input_dir + "/XY_gexp_test_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/pca_scatter_plot_gxpr", "test")
			test_xy_meth_pca, n_comp_PCA = applyDimReduction_PCA(input_dir + "/XY_meth_test_" + str(k) + "_ML_input.tsv", num_of_dim_meth, dirPath_table1_me + "/pca_scatter_plot_meth", "test")
			train_xy_meth_pca, n_comp_PCA = applyDimReduction_PCA(input_dir + "/XY_meth_train_" + str(k) + "_ML_input.tsv", n_comp_PCA, dirPath_table1_me + "/pca_scatter_plot_meth","train")
			train_xy_gxpr_meth_pca = buildIntegratedDataset_DNN(train_xy_gxpr_pca, train_xy_meth_pca, mode)
			test_xy_gxpr_meth_pca = buildIntegratedDataset_DNN(test_xy_gxpr_pca, test_xy_meth_pca, mode)
			dnn_result_output = output_dir + "/[" + str(k) + "]["+ mode + "] DNN_deg_dmg [PCA].tsv"
			doDNN_8(train_xy_gxpr_meth_pca, test_xy_gxpr_meth_pca, dnn_result_output, 1500, "no")



## main
if __name__ == '__main__':
	help_str = "python AD_Prediction_DNN.py" + "\n"

	## input directory
	input_dir_path = "../../results/k_fold_train_test"

	## output directory
	output_dir_path = "../../results/k_fold_train_test_results"
	if not os.path.exists(output_dir_path): os.mkdir(output_dir_path)

	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=str, default=input_dir_path, help=help_str)
	parser.add_argument("--output", type=str, default=output_dir_path, help=help_str)

	args = parser.parse_args()
	main(args)
