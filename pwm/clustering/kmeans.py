import argparse
import sys
import random
import pandas as pd
from sklearn.cluster import KMeans
from gendl import pwm, seqio
from pwm.clustering import clust_lib


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Perform Kmeans Clustering on genetic sequence')
	parser.add_argument('--file1', required=True, type=str,
		metavar='<file>', help='fasta file')
	parser.add_argument('--file0', required=True, type=str,
		metavar='<file>', help='fasta file')
	parser.add_argument('--xvalid', required=False, type=int, default=4,
		metavar='<int>', help='x-fold cross-validation [%(default)s]')
	parser.add_argument('--k', required=False, type=int, default=2,
		metavar='<int>', help='number of clusters for kmeans')
	parser.add_argument('--start', required=False, type=int, default=0,
		metavar='<int>', help='start of the analyzed sequence')
	parser.add_argument('--stop', required=False, type=int, default=42,
		metavar='<int>', help='end of the parsing ')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='end of the analyzed sequence')
	parser.add_argument('--pca', required=False, action='store_true',
		help='perform pca on kmeans')
	arg = parser.parse_args()

	if arg.seed:
		random.seed(arg.seed)

	#read sequences and create a dataframe out of them
	seqs1 = [(1, seq[arg.start:arg.stop]) for name, seq in seqio.read_fasta(arg.file1)]
	seqs0 = [(0, seq[arg.start:arg.stop]) for name, seq in seqio.read_fasta(arg.file0)]
	seqs = seqs1 + seqs0
	random.shuffle(seqs)

	distr_train = []
	distr_test = []

	#splitting data into training and testing

	for train, test in seqio.cross_validation(seqs, arg.xvalid):

		#extracting trues and fakes out of the train data
		train_extr = [seq[1] for seq in train]

		#creating a model using kmeans and train_dff
		traindf = seqio.conv_data(train_extr)
		kmtr = KMeans(arg.k).fit(traindf)

		#regroup seqs to calculate the distribution
		distr_train.append(clust_lib.regroup(kmtr.labels_, train))

		#pca
		if arg.pca:
			clust_lib.pca_kmeans(traindf, train_extr, arg.k)

		#converting test data into df format
		converting = {'A':1.0, 'C':2.0, 'G':3.0, 'T':4.0}
		test_df = []
		for label, seq in test:
			conv_seq = []
			for base in seq:
				if base in converting:
					conv_seq.append(converting[base])
			test_df.append(conv_seq)

		test_df = pd.DataFrame(test_df)

		#running test_df on the kmeans model created with train set
		test_labels = kmtr.predict(test_df)
		#regroup seqs to calculate the distribution
		distr_test.append(clust_lib.regroup(test_labels, test))

	print('Result of K-means clustering TRAIN set: percent of file1 seqs by each cluster')
	clust_lib.outcome(distr_train, arg.k)
	print('Result of K-means clustering TEST set: percent of file1 seqs by each cluster')
	clust_lib.outcome(distr_test, arg.k)
