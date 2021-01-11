import argparse
import sys
import random
import pandas as pd
from sklearn.cluster import KMeans
from io import StringIO
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
	parser.add_argument('--pca', required=False, action='store_true',
		help='perform pca on kmeans training set')
	arg = parser.parse_args()

	if arg.seed:
		random.seed(arg.seed)

	#read sequences and create a dataframe out of them
	seqs1 = [(1, seq[arg.start:arg.stop]) for name, seq in seqio.read_fasta(arg.file1)]
	seqs0 = [(0, seq[arg.start:arg.stop]) for name, seq in seqio.read_fasta(arg.file0)]
	seqs = seqs1 + seqs0
	random.shuffle(seqs)

	percentages = {}
	#splitting data into training and testing
	i = 0
	for train, test in seqio.cross_validation(seqs, arg.xvalid):

		#extracting trues and fakes out of the train data
		trues_train = [seq for label, seq in train if label == 1]
		fakes_train = [seq for label, seq in train if label == 0]

		train_extr = trues_train + fakes_train
		random.shuffle(train_extr)

		#creating a model using kmeans and train_dff
		traindf = clust_lib.conv_data(train_extr)
		kmtr = KMeans(arg.k).fit(traindf)

		#pca
		if arg.pca:
			clust_lib.pca_kmeans(trues_df, trues_train, arg.k1)
			clust_lib.pca_kmeans(fakes_df, fakes_train, arg.k0)

		#converting test data into df format
		converting = {'A':1.0, 'C':2.0, 'G':3.0, 'T':4.0}
		distribution = {}
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

		#predicting labels of test data and show the distribution of each file
		summary = {}
		for label, test in zip(test_labels, test):
			if label not in summary:
				summary[label] = []
				summary[label].append(test)
			else:
				summary[label].append(test)

		for exp_label in summary.keys():
			if exp_label not in distribution:
				distribution[exp_label] = {}
				for obs_label in summary[exp_label]:
					if obs_label[0] not in distribution[exp_label]:
						distribution[exp_label][obs_label[0]] = 1
					else:
						distribution[exp_label][obs_label[0]] += 1
			else:
				for obs_label in summary[exp_label]:
					if obs_label[0] not in distribution[exp_label]:
						distribution[exp_label][obs_label[0]] = 1
					else:
						distribution[exp_label][obs_label[0]] += 1

		print('x-fold:', i)
		for label, seq in distribution.items():
			print('\nKMEANS CLUSTER:', label)
			for label, num in seq.items():
				print('actual label:', label, '\t', 'number of seqs place in cluster:', num)

		print('\n')
		i += 1

