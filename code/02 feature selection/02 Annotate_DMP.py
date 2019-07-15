####################################################################################################################################################
# Annotate_DMP.py
# Author: Chihyun Park
# Email: chihyun.park@yonsei.ac.kr
# Date created: 11/02/2018
# Date lastly modified: 7/1/2019
# Purpose: To assign annotation on the limma results from DMP analysis
# input
#       limma results about DMP
# output
#       annotated results
####################################################################################################################################################
import pandas as pd

def main():
	print("annotate DMP")
	cut_off_logFC = 0.58
	cut_off_pval = 0.05

	final_col_list = ['logFC', 'P.Value', 'UCSC_RefGene_Group', 'UCSC_RefGene_Name']
	TSSlist = ['TSS1500', 'TSS200']

	annot_file = "../../dataset/GPL13534-11288.txt" ## GPL13534-11288
	output_dir = "../../results/k_fold_train_test"

	annot_df = pd.read_csv(annot_file, "\t", header=37)
	annot_df.set_index('ID', inplace=True)
	print(annot_df.ix[:3, :])
	print(annot_df.shape)

	## Training dataset
	for i in range(1, 6):
		input_file = "../../results/k_fold_train_test/DMP/[train " + str(i) + "] AD DMP.tsv"
		dmp_df = pd.read_csv(input_file, sep="\t")
		dmp_annot_df = dmp_df.merge(annot_df, how='left', left_index=True, right_index=True)
		dmp_annot_df = dmp_annot_df[final_col_list]
		dmp_annot_df['UCSC_RefGene_Name'] = dmp_annot_df['UCSC_RefGene_Name'].str.split(';').str[0]
		dmp_annot_df = dmp_annot_df.loc[(abs(dmp_annot_df['logFC']) >= cut_off_logFC) & (dmp_annot_df['P.Value'] < cut_off_pval)]
		dmp_annot_df = dmp_annot_df[dmp_annot_df["UCSC_RefGene_Group"].str.contains('|'.join(TSSlist), na=False)]

		dmp_annot_df.columns = ['LogFC', 'P_value', 'Genomic_Position', 'Gene']
		outfilePath = output_dir + "/XY_meth_train_" + str(i) + "_DMPlist.tsv"
		dmp_annot_df.to_csv(outfilePath, header=True, index=True, sep="\t")

	## Training dataset
	for i in range(1, 6):
		input_file = "../../results/k_fold_train_test/DMP/[test " + str(i) + "] AD DMP.tsv"
		dmp_df = pd.read_csv(input_file, sep="\t")
		dmp_annot_df = dmp_df.merge(annot_df, how='left', left_index=True, right_index=True)
		dmp_annot_df = dmp_annot_df[final_col_list]
		dmp_annot_df['UCSC_RefGene_Name'] = dmp_annot_df['UCSC_RefGene_Name'].str.split(';').str[0]
		dmp_annot_df = dmp_annot_df.loc[
			(abs(dmp_annot_df['logFC']) >= cut_off_logFC) & (dmp_annot_df['P.Value'] < cut_off_pval)]
		dmp_annot_df = dmp_annot_df[dmp_annot_df["UCSC_RefGene_Group"].str.contains('|'.join(TSSlist), na=False)]

		dmp_annot_df.columns = ['LogFC', 'P_value', 'Genomic_Position', 'Gene']
		outfilePath = output_dir + "/XY_meth_test_" + str(i) + "_DMPlist.tsv"
		dmp_annot_df.to_csv(outfilePath, header=True, index=True, sep="\t")


## main
if __name__ == '__main__':
	main()