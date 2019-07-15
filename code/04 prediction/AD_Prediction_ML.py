####################################################################################################################################################
# AD_Prediction_by_ML.py
# Author: Chihyun Park
# Email: chihyun.park@yonsei.ac.kr
# Date created: 11/02/2018
# Date lastly modified: 7/1/2019
# Purpose: To construct AD prediction model using conventional ML approach
# input
#       for each K fold dataset
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
import random as rd
import csv
import pandas as pd
import argparse
import os
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


## integrate multi-omics datasets
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

	print("[limma - DEG] Number of gene set: " + str(len(geneSet)))
	#print("geneSet: " + str(geneSet))

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
		x_tmpRow = xy_me_ge_values[i, 1:colSize - 2]
		y_tmpRow = xy_me_ge_values[i, colSize - 1:colSize]

		x_data_List.append(x_tmpRow)
		y_data_List.append(y_tmpRow)

	return np.array(x_data_List), np.array(y_data_List)


## perform conventional machine learning
def doMachineLearning_single_Kfold(xy_train, xy_test, k):
	print("doMachineLearning_single")

	x_train, y_train = partitionTrainTest_ML_for_CV_DNN(xy_train)
	x_test, y_test = partitionTrainTest_ML_for_CV_DNN(xy_test)

	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)
	y_train = y_train.astype(np.int)
	y_test = y_test.astype(np.int)

	"""
	sc = StandardScaler()
	sc.fit(x_train)
	x_train = sc.transform(x_train)
	sc.fit(x_test)
	x_test = sc.transform(x_test)
	"""

	print("## Random Forest")
	###############################################################################################
	# define a model #rdf_clf.fit(x_data_std, y_data)
	rdf_clf = RandomForestClassifier(criterion='entropy', oob_score=True, n_estimators=100, n_jobs=-1, random_state=0, max_depth=6)
	rdf_clf.fit(x_train, y_train)

	predicted = rdf_clf.predict(x_test)
	test_acc = accuracy_score(y_test, predicted)
	training_acc = accuracy_score(y_train, rdf_clf.predict(x_train))
	roc_auc = metrics.roc_auc_score(y_test.ravel(), predicted.ravel())

	print(str(k) + "-fold training accuracy:" + "\t" + str(training_acc))
	print(str(k) + "-fold test accuracy:" + "\t" + str(test_acc))
	print(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))
	rd_acc = test_acc
	rd_auc = roc_auc

	# SVM ###############################################################################################
	print("## SVM")
	###############################################################################################
	# define a model #rdf_clf.fit(x_data_std, y_data)
	svm_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
					  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
					  max_iter=-1, probability=True, random_state=None, shrinking=True,
					  tol=0.001, verbose=False)
	svm_clf.fit(x_train, y_train)

	predicted = svm_clf.predict(x_test)
	test_acc = accuracy_score(y_test, predicted)
	training_acc = accuracy_score(y_train, svm_clf.predict(x_train))
	roc_auc = metrics.roc_auc_score(y_test.ravel(), predicted.ravel())

	print(str(k) + "-fold training accuracy:" + "\t" + str(training_acc))
	print(str(k) + "-fold test accuracy:" + "\t" + str(test_acc))
	print(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))
	svm_acc = test_acc
	svm_auc = roc_auc
	###############################################################################################


	# naive bayesian classifier
	print("## naive bayesian")
	###############################################################################################
	# define a model #rdf_clf.fit(x_data_std, y_data)
	gnb_clf = GaussianNB()
	gnb_clf.fit(x_train, y_train)

	predicted = gnb_clf.predict(x_test)
	test_acc = accuracy_score(y_test, predicted)
	training_acc = accuracy_score(y_train, gnb_clf.predict(x_train))
	roc_auc = metrics.roc_auc_score(y_test.ravel(), predicted.ravel())

	print(str(k) + "-fold training accuracy:" + "\t" + str(training_acc))
	print(str(k) + "-fold test accuracy:" + "\t" + str(test_acc))
	print(str(k) + "-fold test roc_auc:" + "\t" + str(roc_auc))
	nb_acc = test_acc
	nb_auc = roc_auc
	###############################################################################################

	return rd_acc, rd_auc, svm_acc, svm_auc, nb_acc, nb_auc


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
	print("Machine Learning approach")
	input_dir = args.input ## ./results/k_fold_train_test
	output_dir = args.output ## ./results/k_fold_train_test_results

	for j in range(0, 2):
		if j == 0:
			mode = "balanced"  # "unbalanced"
		else:
			mode = "unbalanced"  # "unbalanced"

		print("mode: " + mode)

		## number of features
		num_deg_ge_List =[]
		num_dmg_me_List =[]

		num_tsne_ge_List =[]
		num_tsne_me_List =[]

		num_pca_ge_List =[]
		num_pca_me_List =[]

		## results
		deg_dmg_ge_me_DNN_acc_List = []

		deg_ge_RF_acc_List =[]
		deg_ge_RF_auc_List =[]
		dmg_me_RF_acc_List =[]
		dmg_me_RF_auc_List =[]
		deg_dmg_ge_me_RF_acc_List =[]
		deg_dmg_ge_me_RF_auc_List =[]

		deg_ge_SVM_acc_List =[]
		deg_ge_SVM_auc_List =[]
		dmg_me_SVM_acc_List =[]
		dmg_me_SVM_auc_List =[]
		deg_dmg_ge_me_SVM_acc_List =[]
		deg_dmg_ge_me_SVM_auc_List =[]

		deg_ge_NB_acc_List =[]
		deg_ge_NB_auc_List =[]
		dmg_me_NB_acc_List =[]
		dmg_me_NB_auc_List =[]
		deg_dmg_ge_me_NB_acc_List =[]
		deg_dmg_ge_me_NB_auc_List =[]

		tsne_ge_RF_acc_List =[]
		tsne_ge_RF_auc_List =[]
		tsne_me_RF_acc_List =[]
		tsne_me_RF_auc_List =[]
		tsne_ge_me_RF_acc_List =[]
		tsne_ge_me_RF_auc_List =[]

		tsne_ge_SVM_acc_List =[]
		tsne_ge_SVM_auc_List =[]
		tsne_me_SVM_acc_List =[]
		tsne_me_SVM_auc_List =[]
		tsne_ge_me_SVM_acc_List =[]
		tsne_ge_me_SVM_auc_List =[]

		tsne_ge_NB_acc_List =[]
		tsne_ge_NB_auc_List =[]
		tsne_me_NB_acc_List =[]
		tsne_me_NB_auc_List =[]
		tsne_ge_me_NB_acc_List =[]
		tsne_ge_me_NB_auc_List =[]

		pca_ge_RF_acc_List =[]
		pca_ge_RF_auc_List =[]
		pca_me_RF_acc_List =[]
		pca_me_RF_auc_List =[]
		pca_ge_me_RF_acc_List =[]
		pca_ge_me_RF_auc_List =[]

		pca_ge_SVM_acc_List =[]
		pca_ge_SVM_auc_List =[]
		pca_me_SVM_acc_List =[]
		pca_me_SVM_auc_List =[]
		pca_ge_me_SVM_acc_List =[]
		pca_ge_me_SVM_auc_List =[]

		pca_ge_NB_acc_List =[]
		pca_ge_NB_auc_List =[]
		pca_me_NB_acc_List =[]
		pca_me_NB_auc_List =[]
		pca_ge_me_NB_acc_List =[]
		pca_ge_me_NB_auc_List =[]

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


			## ExpResult 1. PCA, tSNE + ML algorithms with same # of reduced features for each GE, Meth
			## ExpResult 2. PCA, tSNE + ML algorithms with same # of reduced features by integrating GE, Meth
			## ExpResult 3. DEG, DMG, DEG + DMG + ML algorithms
			## ExpResult 4. DEG + DMG + DNN

			print("\n\nExperiment 3. DEG, DMP + ML")
			## ExpResult 3. DEG, DMG, DEG + DMG + ML algorithms
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

			train_xy_gxpr = applyFeatSel_DEG_intersectGene(input_dir + "/XY_gexp_train_" + str(k) + "_ML_input.tsv", its_geneSet)
			train_xy_meth = applyFeatSel_DMP_intersectGene(input_dir + "/XY_meth_train_" + str(k) + "_ML_input.tsv", its_geneSet, geneCpgSet_map)

			print("train_xy_gxpr: " + str(train_xy_gxpr.shape))
			print("train_xy_meth: " + str(train_xy_meth.shape))

			print("\n")
			print("# feature (gene expression):" + str(train_xy_gxpr.shape[1]))
			print("# feature (DNA methylation):" + str(train_xy_meth.shape[1]))
			print("\n")

			num_deg_ge = train_xy_gxpr.shape[1] - 2
			num_dmg_me = train_xy_meth.shape[1] - 2

			test_xy_gxpr = applyFeatSel_DEG_intersectGene(input_dir + "/XY_gexp_test_" + str(k) + "_ML_input.tsv", its_geneSet)
			test_xy_meth = applyFeatSel_DMP_intersectGene(input_dir + "/XY_meth_test_" + str(k) + "_ML_input.tsv", its_geneSet, geneCpgSet_map)

			print("test_xy_gxpr: " + str(test_xy_gxpr.shape))
			print("test_xy_meth: " + str(test_xy_meth.shape))

			train_xy_gxpr_meth = buildIntegratedDataset_DNN(train_xy_gxpr, train_xy_meth, mode) # "unbalanced"
			test_xy_gxpr_meth = buildIntegratedDataset_DNN(test_xy_gxpr, test_xy_meth, mode)

			print("train_xy_gxpr_meth: " + str(train_xy_gxpr_meth.shape))
			print("test_xy_gxpr_meth: " + str(test_xy_gxpr_meth.shape))

			deg_ge_RF_acc, deg_ge_RF_auc, deg_ge_SVM_acc, deg_ge_SVM_auc, deg_ge_NB_acc, deg_ge_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr, test_xy_gxpr, k)
			dmg_me_RF_acc, dmg_me_RF_auc, dmg_me_SVM_acc, dmg_me_SVM_auc, dmg_me_NB_acc, dmg_me_NB_auc = doMachineLearning_single_Kfold(train_xy_meth, test_xy_meth, k)
			deg_dmg_ge_me_RF_acc, deg_dmg_ge_me_RF_auc, deg_dmg_ge_me_SVM_acc, deg_dmg_ge_me_SVM_auc, deg_dmg_ge_me_NB_acc, deg_dmg_ge_me_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr_meth, test_xy_gxpr_meth, k)

			print("\n\nExperiment 1~2. PCA, t-SNE + ML")
			num_of_dim_gxpr = train_xy_gxpr.shape[1] - 2
			num_of_dim_meth = train_xy_meth.shape[1] - 2
			num_of_dim_gexp_meth = train_xy_gxpr_meth.shape[1] - 2

			print("num_of_dim_gxpr: " + str(num_of_dim_gxpr))
			print("num_of_dim_meth: " + str(num_of_dim_meth))
			print("num_of_dim_gexp_meth: " + str(num_of_dim_gexp_meth))


			## ExpResult 1. PCA, tSNE + ML algorithms with same # of reduced features for each GE, Meth
			## ExpResult 2. PCA, tSNE + ML algorithms with same # of reduced features by integrating GE, Meth
			## dimension reduction by t-SNE, PCA should be done using training dataset
			## 1.1 t-SNE gexpr, meth
			train_xy_gxpr_tsne = applyDimReduction_TSNE(input_dir + "/XY_gexp_train_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/tsne_scatter_plot_gxpr", "train")
			test_xy_gxpr_tsne = applyDimReduction_TSNE(input_dir + "/XY_gexp_test_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/tsne_scatter_plot_gxpr", "test")
			train_xy_meth_tsne = applyDimReduction_TSNE(input_dir + "/XY_meth_train_" + str(k) + "_ML_input.tsv", num_of_dim_meth, dirPath_table1_me + "/tsne_scatter_plot_meth", "train")
			test_xy_meth_tsne = applyDimReduction_TSNE(input_dir + "/XY_meth_test_" + str(k) + "_ML_input.tsv", num_of_dim_meth, dirPath_table1_me + "/tsne_scatter_plot_meth", "test")

			num_tsne_ge = train_xy_gxpr_tsne.shape[1] - 2
			num_tsne_me = train_xy_meth_tsne.shape[1] - 2

			train_xy_gxpr_meth_tsne = buildIntegratedDataset_DNN(train_xy_gxpr_tsne, train_xy_meth_tsne, mode)
			test_xy_gxpr_meth_tsne = buildIntegratedDataset_DNN(test_xy_gxpr_tsne, test_xy_meth_tsne, mode)

			tsne_ge_RF_acc, tsne_ge_RF_auc, tsne_ge_SVM_acc, tsne_ge_SVM_auc, tsne_ge_NB_acc, tsne_ge_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr_tsne, test_xy_gxpr_tsne, k)
			tsne_me_RF_acc, tsne_me_RF_auc, tsne_me_SVM_acc, tsne_me_SVM_auc, tsne_me_NB_acc, tsne_me_NB_auc = doMachineLearning_single_Kfold(train_xy_meth_tsne, test_xy_meth_tsne, k)
			tsne_ge_me_RF_acc, tsne_ge_me_RF_auc, tsne_ge_me_SVM_acc, tsne_ge_me_SVM_auc, tsne_ge_me_NB_acc, tsne_ge_me_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr_meth_tsne, test_xy_gxpr_meth_tsne, k)

			## 1.2 PCA gexpr, meth
			train_xy_gxpr_pca, _ = applyDimReduction_PCA(input_dir + "/XY_gexp_train_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/pca_scatter_plot_gxpr", "train")
			test_xy_gxpr_pca, _ = applyDimReduction_PCA(input_dir + "/XY_gexp_test_" + str(k) + "_ML_input.tsv", num_of_dim_gxpr, dirPath_table1_ge + "/pca_scatter_plot_gxpr", "test")
			test_xy_meth_pca, n_comp_PCA = applyDimReduction_PCA(input_dir + "/XY_meth_test_" + str(k) + "_ML_input.tsv", num_of_dim_meth, dirPath_table1_me + "/pca_scatter_plot_meth", "test")
			train_xy_meth_pca, n_comp_PCA = applyDimReduction_PCA(input_dir + "/XY_meth_train_" + str(k) + "_ML_input.tsv", n_comp_PCA, dirPath_table1_me + "/pca_scatter_plot_meth", "train")
			print("n_comp_PCA: " + str(n_comp_PCA))

			num_pca_ge = train_xy_gxpr_pca.shape[1] - 2
			num_pca_me = train_xy_meth_pca.shape[1] - 2

			train_xy_gxpr_meth_pca = buildIntegratedDataset_DNN(train_xy_gxpr_pca, train_xy_meth_pca, mode)
			test_xy_gxpr_meth_pca = buildIntegratedDataset_DNN(test_xy_gxpr_pca, test_xy_meth_pca, mode)

			pca_ge_RF_acc, pca_ge_RF_auc, pca_ge_SVM_acc, pca_ge_SVM_auc, pca_ge_NB_acc, pca_ge_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr_pca, test_xy_gxpr_pca, k)
			pca_me_RF_acc, pca_me_RF_auc, pca_me_SVM_acc, pca_me_SVM_auc, pca_me_NB_acc, pca_me_NB_auc = doMachineLearning_single_Kfold(train_xy_meth_pca, test_xy_meth_pca, k)
			pca_ge_me_RF_acc, pca_ge_me_RF_auc, pca_ge_me_SVM_acc, pca_ge_me_SVM_auc, pca_ge_me_NB_acc, pca_ge_me_NB_auc = doMachineLearning_single_Kfold(train_xy_gxpr_meth_pca, test_xy_gxpr_meth_pca, k)

			## save the results
			num_deg_ge_List.append(num_deg_ge)
			num_dmg_me_List.append(num_dmg_me)

			num_tsne_ge_List.append(num_tsne_ge)
			num_tsne_me_List.append(num_tsne_me)

			num_pca_ge_List.append(num_pca_ge)
			num_pca_me_List.append(num_pca_me)

			## results
			deg_ge_RF_acc_List.append(deg_ge_RF_acc)
			deg_ge_RF_auc_List.append(deg_ge_RF_auc)
			dmg_me_RF_acc_List.append(dmg_me_RF_acc)
			dmg_me_RF_auc_List.append(dmg_me_RF_auc)
			deg_dmg_ge_me_RF_acc_List.append(deg_dmg_ge_me_RF_acc)
			deg_dmg_ge_me_RF_auc_List.append(deg_dmg_ge_me_RF_auc)

			deg_ge_SVM_acc_List.append(deg_ge_SVM_acc)
			deg_ge_SVM_auc_List.append(deg_ge_SVM_auc)
			dmg_me_SVM_acc_List.append(dmg_me_SVM_acc)
			dmg_me_SVM_auc_List.append(dmg_me_SVM_auc)
			deg_dmg_ge_me_SVM_acc_List.append(deg_dmg_ge_me_SVM_acc)
			deg_dmg_ge_me_SVM_auc_List.append(deg_dmg_ge_me_SVM_auc)

			deg_ge_NB_acc_List.append(deg_ge_NB_acc)
			deg_ge_NB_auc_List.append(deg_ge_NB_auc)
			dmg_me_NB_acc_List.append(dmg_me_NB_acc)
			dmg_me_NB_auc_List.append(dmg_me_NB_auc)
			deg_dmg_ge_me_NB_acc_List.append(deg_dmg_ge_me_NB_acc)
			deg_dmg_ge_me_NB_auc_List.append(deg_dmg_ge_me_NB_auc)

			tsne_ge_RF_acc_List.append(tsne_ge_RF_acc)
			tsne_ge_RF_auc_List.append(tsne_ge_RF_auc)
			tsne_me_RF_acc_List.append(tsne_me_RF_acc)
			tsne_me_RF_auc_List.append(tsne_me_RF_auc)
			tsne_ge_me_RF_acc_List.append(tsne_ge_me_RF_acc)
			tsne_ge_me_RF_auc_List.append(tsne_ge_me_RF_auc)

			tsne_ge_SVM_acc_List.append(tsne_ge_SVM_acc)
			tsne_ge_SVM_auc_List.append(tsne_ge_SVM_auc)
			tsne_me_SVM_acc_List.append(tsne_me_SVM_acc)
			tsne_me_SVM_auc_List.append(tsne_me_SVM_auc)
			tsne_ge_me_SVM_acc_List.append(tsne_ge_me_SVM_acc)
			tsne_ge_me_SVM_auc_List.append(tsne_ge_me_SVM_auc)

			tsne_ge_NB_acc_List.append(tsne_ge_NB_acc)
			tsne_ge_NB_auc_List.append(tsne_ge_NB_auc)
			tsne_me_NB_acc_List.append(tsne_me_NB_acc)
			tsne_me_NB_auc_List.append(tsne_me_NB_auc)
			tsne_ge_me_NB_acc_List.append(tsne_ge_me_NB_acc)
			tsne_ge_me_NB_auc_List.append(tsne_ge_me_NB_auc)

			pca_ge_RF_acc_List.append(pca_ge_RF_acc)
			pca_ge_RF_auc_List.append(pca_ge_RF_auc)
			pca_me_RF_acc_List.append(pca_me_RF_acc)
			pca_me_RF_auc_List.append(pca_me_RF_auc)
			pca_ge_me_RF_acc_List.append(pca_ge_me_RF_acc)
			pca_ge_me_RF_auc_List.append(pca_ge_me_RF_auc)

			pca_ge_SVM_acc_List.append(pca_ge_SVM_acc)
			pca_ge_SVM_auc_List.append(pca_ge_SVM_auc)
			pca_me_SVM_acc_List.append(pca_me_SVM_acc)
			pca_me_SVM_auc_List.append(pca_me_SVM_auc)
			pca_ge_me_SVM_acc_List.append(pca_ge_me_SVM_acc)
			pca_ge_me_SVM_auc_List.append(pca_ge_me_SVM_auc)

			pca_ge_NB_acc_List.append(pca_ge_NB_acc)
			pca_ge_NB_auc_List.append(pca_ge_NB_auc)
			pca_me_NB_acc_List.append(pca_me_NB_acc)
			pca_me_NB_auc_List.append(pca_me_NB_auc)
			pca_ge_me_NB_acc_List.append(pca_ge_me_NB_acc)
			pca_ge_me_NB_auc_List.append(pca_ge_me_NB_auc)


		res_dict = {}
		res_dict["num_deg_ge"] = num_deg_ge_List
		res_dict["num_dmg_me"] = num_dmg_me_List
		res_dict["num_tsne_ge"] = num_tsne_ge_List
		res_dict["num_tsne_me"] = num_tsne_me_List
		res_dict["num_pca_ge"] = num_pca_ge_List
		res_dict["num_pca_me"] = num_pca_me_List

		res_dict["deg_ge_RF_acc"] = deg_ge_RF_acc_List
		res_dict["deg_ge_RF_auc"] = deg_ge_RF_auc_List
		res_dict["dmg_me_RF_acc"] = dmg_me_RF_acc_List
		res_dict["dmg_me_RF_auc"] = dmg_me_RF_auc_List
		res_dict["deg_dmg_ge_me_RF_acc"] = deg_dmg_ge_me_RF_acc_List
		res_dict["deg_dmg_ge_me_RF_auc"] = deg_dmg_ge_me_RF_auc_List

		res_dict["deg_ge_SVM_acc"] = deg_ge_SVM_acc_List
		res_dict["deg_ge_SVM_auc"] = deg_ge_SVM_auc_List
		res_dict["dmg_me_SVM_acc"] = dmg_me_SVM_acc_List
		res_dict["dmg_me_SVM_auc"] = dmg_me_SVM_auc_List
		res_dict["deg_dmg_ge_me_SVM_acc"] = deg_dmg_ge_me_SVM_acc_List
		res_dict["deg_dmg_ge_me_SVM_auc"] = deg_dmg_ge_me_SVM_auc_List

		res_dict["deg_ge_NB_acc"] = deg_ge_NB_acc_List
		res_dict["deg_ge_NB_auc"] = deg_ge_NB_auc_List
		res_dict["dmg_me_NB_acc"] = dmg_me_NB_acc_List
		res_dict["dmg_me_NB_auc"] = dmg_me_NB_auc_List
		res_dict["deg_dmg_ge_me_NB_acc"] = deg_dmg_ge_me_NB_acc_List
		res_dict["deg_dmg_ge_me_NB_auc"] = deg_dmg_ge_me_NB_auc_List

		res_dict["tsne_ge_RF_acc"] = tsne_ge_RF_acc_List
		res_dict["tsne_ge_RF_auc"] = tsne_ge_RF_auc_List
		res_dict["tsne_me_RF_acc"] = tsne_me_RF_acc_List
		res_dict["tsne_me_RF_auc"] = tsne_me_RF_auc_List
		res_dict["tsne_ge_me_RF_acc"] = tsne_ge_me_RF_acc_List
		res_dict["tsne_ge_me_RF_auc"] = tsne_ge_me_RF_auc_List

		res_dict["tsne_ge_SVM_acc"] = tsne_ge_SVM_acc_List
		res_dict["tsne_ge_SVM_auc"] = tsne_ge_SVM_auc_List
		res_dict["tsne_me_SVM_acc"] = tsne_me_SVM_acc_List
		res_dict["tsne_me_SVM_auc"] = tsne_me_SVM_auc_List
		res_dict["tsne_ge_me_SVM_acc"] = tsne_ge_me_SVM_acc_List
		res_dict["tsne_ge_me_SVM_auc"] = tsne_ge_me_SVM_auc_List

		res_dict["tsne_ge_NB_acc"] = tsne_ge_NB_acc_List
		res_dict["tsne_ge_NB_auc"] = tsne_ge_NB_auc_List
		res_dict["tsne_me_NB_acc"] = tsne_me_NB_acc_List
		res_dict["tsne_me_NB_auc"] = tsne_me_NB_auc_List
		res_dict["tsne_ge_me_NB_acc"] = tsne_ge_me_NB_acc_List
		res_dict["tsne_ge_me_NB_auc"] = tsne_ge_me_NB_auc_List

		res_dict["pca_ge_RF_acc"] = pca_ge_RF_acc_List
		res_dict["pca_ge_RF_auc"] = pca_ge_RF_auc_List
		res_dict["pca_me_RF_acc"] = pca_me_RF_acc_List
		res_dict["pca_me_RF_auc"] = pca_me_RF_auc_List
		res_dict["pca_ge_me_RF_acc"] = pca_ge_me_RF_acc_List
		res_dict["pca_ge_me_RF_auc"] = pca_ge_me_RF_auc_List

		res_dict["pca_ge_SVM_acc"] = pca_ge_SVM_acc_List
		res_dict["pca_ge_SVM_auc"] = pca_ge_SVM_auc_List
		res_dict["pca_me_SVM_acc"] = pca_me_SVM_acc_List
		res_dict["pca_me_SVM_auc"] = pca_me_SVM_auc_List
		res_dict["pca_ge_me_SVM_acc"] = pca_ge_me_SVM_acc_List
		res_dict["pca_ge_me_SVM_auc"] = pca_ge_me_SVM_auc_List

		res_dict["pca_ge_NB_acc"] = pca_ge_NB_acc_List
		res_dict["pca_ge_NB_auc"] = pca_ge_NB_auc_List
		res_dict["pca_me_NB_acc"] = pca_me_NB_acc_List
		res_dict["pca_me_NB_auc"] = pca_me_NB_auc_List
		res_dict["pca_ge_me_NB_acc"] = pca_ge_me_NB_acc_List
		res_dict["pca_ge_me_NB_auc"] = pca_ge_me_NB_auc_List

		all_df = pd.DataFrame(res_dict)

		columns_acc = ["num_deg_ge", "deg_ge_RF_acc", "deg_ge_SVM_acc", "deg_ge_NB_acc",
					   "num_dmg_me", "dmg_me_RF_acc", "dmg_me_SVM_acc", "dmg_me_NB_acc",
					   "deg_dmg_ge_me_RF_acc", "deg_dmg_ge_me_SVM_acc", "deg_dmg_ge_me_NB_acc",

					   "num_tsne_ge", "tsne_ge_RF_acc", "tsne_ge_SVM_acc", "tsne_ge_NB_acc",
					   "num_tsne_me", "tsne_me_RF_acc", "tsne_me_SVM_acc", "tsne_me_NB_acc",
					   "tsne_ge_me_RF_acc", "tsne_ge_me_SVM_acc", "tsne_ge_me_NB_acc",

					   "num_pca_ge", "pca_ge_RF_acc", "pca_ge_SVM_acc", "pca_ge_NB_acc",
					   "num_pca_me", "pca_me_RF_acc", "pca_me_SVM_acc", "pca_me_NB_acc",
					   "pca_ge_me_RF_acc", "pca_ge_me_SVM_acc", "pca_ge_me_NB_acc"]
		acc_df = all_df[columns_acc]
		print("acc_df: " + str(acc_df.shape))

		columns_auc = ["num_deg_ge", "deg_ge_RF_auc", "deg_ge_SVM_auc", "deg_ge_NB_auc",
					   "num_dmg_me", "dmg_me_RF_auc", "dmg_me_SVM_auc", "dmg_me_NB_auc",
					   "deg_dmg_ge_me_RF_auc", "deg_dmg_ge_me_SVM_auc", "deg_dmg_ge_me_NB_auc",

					   "num_tsne_ge", "tsne_ge_RF_auc", "tsne_ge_SVM_auc", "tsne_ge_NB_auc",
					   "num_tsne_me", "tsne_me_RF_auc", "tsne_me_SVM_auc", "tsne_me_NB_auc",
					   "tsne_ge_me_RF_auc", "tsne_ge_me_SVM_auc", "tsne_ge_me_NB_auc",

					   "num_pca_ge", "pca_ge_RF_auc", "pca_ge_SVM_auc", "pca_ge_NB_auc",
					   "num_pca_me", "pca_me_RF_auc", "pca_me_SVM_auc", "pca_me_NB_auc",
					   "pca_ge_me_RF_auc", "pca_ge_me_SVM_auc", "pca_ge_me_NB_auc"]
		auc_df = all_df[columns_auc]
		print("auc_df: " + str(auc_df.shape))

		acc_df.to_csv(output_dir + "/["+ mode + "] ML acc_k1-5.tsv", header=True, index=False, sep="\t")
		auc_df.to_csv(output_dir + "/["+ mode + "] ML auc_k1-5.tsv", header=True, index=False, sep="\t")


## main
if __name__ == '__main__':
	help_str = "python AD_Prediction_ML.py" + "\n"

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
