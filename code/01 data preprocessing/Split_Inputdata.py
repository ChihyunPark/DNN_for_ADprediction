####################################################################################################################################################
# Split_Inputdata.py
# Author: Chihyun Park
# Email: chihyun.park@yonsei.ac.kr
# Date created: 7/1/2019
# Date lastly modified: 7/1/2019
# Purpose: To split raw input files using k-fold cross validation
# input
#       1. gene expression (samples x genes) with label (AD, Normal)
#       2. DNA methylation (samples x CpG probes) with label (AD, Normal)
# output
#       splitted gene expression, DNA methylation dataset
####################################################################################################################################################

import argparse
import pandas as pd
import os
from sklearn.model_selection import KFold


def main(args):
	print("main")
	input_gexp = args.input_1
	input_meth = args.input_2
	output_dir = args.output

	## divide entire into training / test set
	k = 5
	kf = KFold(k, True, 1)

	## load input gene expression, DNAmethylation
	gexp_all_df = pd.read_csv(input_gexp, delimiter="\t")
	#gexp_all_df.set_index('SampleID', inplace=True)
	print("gexp_all_df: " + str(gexp_all_df.shape))
	print(gexp_all_df.ix[:7,:7])

	col_list = gexp_all_df.columns.values.tolist()
	gene_list = col_list[1:-2]
	print(gene_list[:10])
	print("gene list size: " + str(len(gene_list)))

	## gene expression
	kiter = 1
	for train, test in kf.split(gexp_all_df):
		print("K: " + str(kiter))

		XY_gexp_train_df = gexp_all_df.iloc[train]
		XY_gexp_test_df = gexp_all_df.iloc[test]

		## make machine learning input
		outfilePath = output_dir + "/XY_gexp_train_" + str(kiter) + "_ML_input.tsv"
		XY_gexp_train_df.to_csv(outfilePath, header=True, index=False, sep="\t")
		outfilePath = output_dir + "/XY_gexp_test_" + str(kiter) + "_ML_input.tsv"
		XY_gexp_test_df.to_csv(outfilePath, header=True, index=False, sep="\t")


		print(XY_gexp_train_df.ix[:5, :5])
		print("XY_gexp_train_df: " + str(XY_gexp_train_df.shape))

		print(XY_gexp_test_df.ix[:5, :5])
		print("XY_gexp_test_df: " + str(XY_gexp_test_df.shape))

		## make output data
		#XY_gexp_train_df['Label'] = np.where(XY_gexp_train_df['Label_AD']==1, 'AD', 'Normal')
		col_list = ['SampleID']
		# col_list.append('Label')
		col_list.append('Label_AD')
		col_list.append('Label_No')
		col_list.extend(gene_list)
		XY_gexp_train_out_df = XY_gexp_train_df[col_list]
		print("XY_gexp_train_out_df: " + str(XY_gexp_train_out_df.shape))
		outfilePath = output_dir + "/XY_gexp_train_" + str(kiter) + ".tsv"
		XY_gexp_train_out_df.T.to_csv(outfilePath, header=False, index=True, sep="\t")

		## make output data
		#XY_gexp_test_df['Label'] = np.where(XY_gexp_test_df['Label_AD'] == 1, 'AD', 'Normal')
		col_list = ['SampleID']
		# col_list.append('Label')
		col_list.append('Label_AD')
		col_list.append('Label_No')
		col_list.extend(gene_list)
		XY_gexp_test_out_df = XY_gexp_test_df[col_list]
		print("XY_gexp_test_out_df: " + str(XY_gexp_test_out_df.shape))
		outfilePath = output_dir + "/XY_gexp_test_" + str(kiter) + ".tsv"
		XY_gexp_test_out_df.T.to_csv(outfilePath, header=False, index=True, sep="\t")

		kiter += 1



	meth_all_df = pd.read_csv(input_meth, delimiter="\t")
	#meth_all_df.set_index('SampleID', inplace=True)
	print("meth_all_df: " + str(meth_all_df.shape))
	print(meth_all_df.ix[:7, :7])

	col_list = meth_all_df.columns.values.tolist()
	mpro_list = col_list[1:-2]
	print(mpro_list[:10])
	print("Probe list size: " + str(len(mpro_list)))

	## gene expression
	kiter = 1
	for train, test in kf.split(meth_all_df):
		print("K: " + str(kiter))

		XY_meth_train_df = meth_all_df.iloc[train]
		XY_meth_test_df = meth_all_df.iloc[test]

		## make machine learning input
		outfilePath = output_dir + "/XY_meth_train_" + str(kiter) + "_ML_input.tsv"
		XY_meth_train_df.to_csv(outfilePath, header=True, index=False, sep="\t")
		outfilePath = output_dir + "/XY_meth_test_" + str(kiter) + "_ML_input.tsv"
		XY_meth_test_df.to_csv(outfilePath, header=True, index=False, sep="\t")

		print(XY_meth_train_df.ix[:5, :5])
		print("XY_meth_train_df: " + str(XY_meth_train_df.shape))

		print(XY_meth_test_df.ix[:5, :5])
		print("XY_meth_test_df: " + str(XY_meth_test_df.shape))

		## make output data
		#XY_meth_train_df['Label'].loc[XY_meth_train_df["Label_AD"] == 1] = "AD"
		#XY_meth_train_df['Label'].loc[XY_meth_train_df["Label_AD"] == 0] = "Normal"
		#XY_meth_train_df['Label'] = np.where(XY_meth_train_df['Label_AD'] == 1, 'AD', 'Normal')
		col_list = ['SampleID']
		#col_list.append('Label')
		col_list.append('Label_AD')
		col_list.append('Label_No')
		col_list.extend(mpro_list)
		XY_meth_train_out_df = XY_meth_train_df[col_list]
		print("XY_meth_train_out_df: " + str(XY_meth_train_out_df.shape))
		outfilePath = output_dir + "/XY_meth_train_" + str(kiter) + ".tsv"
		XY_meth_train_out_df.T.to_csv(outfilePath, header=False, index=True, sep="\t")

		## make output data
		#XY_meth_test_df['Label'].loc[XY_meth_test_df["Label_AD"] == 1] = "AD"
		#XY_meth_test_df['Label'].loc[XY_meth_test_df["Label_AD"] == 0] = "Normal"
		#XY_meth_test_df['Label'] = np.where(XY_meth_test_df['Label_AD'] == 1, 'AD', 'Normal')
		col_list = ['SampleID']
		# col_list.append('Label')
		col_list.append('Label_AD')
		col_list.append('Label_No')
		col_list.extend(mpro_list)
		XY_meth_test_out_df = XY_meth_test_df[col_list]
		print("XY_meth_test_out_df: " + str(XY_meth_test_out_df.shape))
		outfilePath = output_dir + "/XY_meth_test_" + str(kiter) + ".tsv"
		XY_meth_test_out_df.T.to_csv(outfilePath, header=False, index=True, sep="\t")

		kiter += 1




## main
if __name__ == '__main__':
	help_str = "python Split_Inputdata.py" + "\n"

	## input files
	## file 1: gene expression (samples x genes) with label (AD, Normal)
	## file 2: DNA methylation (samples x CpG probes) with label (AD, Normal)
	input_file_geneExpr = "../../dataset/allforDNN_ge.txt"
	input_file_DNAMeth = "../../dataset/allforDNN_me.txt"

	## output directory
	## performance of the model
	output_dir = "../../results/k_fold_train_test"
	if not os.path.exists(output_dir): os.mkdir(output_dir)

	parser = argparse.ArgumentParser()
	parser.add_argument("--input_1", type=str, default=input_file_geneExpr, help=help_str)
	parser.add_argument("--input_2", type=str, default=input_file_DNAMeth, help=help_str)
	parser.add_argument("--output", type=str, default=output_dir, help=help_str)

	args = parser.parse_args()
	main(args)


