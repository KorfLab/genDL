import argparse
import sys
import random
from gendl import seqio, pwm
from pwm.clustering import clust_lib
from sklearn.cluster import KMeans

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Perform Kmeans Clustering on genetic sequence')
	parser.add_argument('--file1', required=True, type=str,
		metavar='<file>', help='fasta file1')
	parser.add_argument('--file0', required=True, type=str,
		metavar='<file>', help='fasta file0')
	parser.add_argument('--xvalid', required=False, type=int, default=4,
		metavar='<int>', help='x-fold cross-validation [%(default)s]')
	parser.add_argument('--k1', required=False, type=int, default=2,
		metavar='<int>', help='number of clusters for file 1')
	parser.add_argument('--k0', required=False, type=int, default=2,
		metavar='<int>', help='number of clusters for file 0')
	parser.add_argument('--start', required=False, type=int, default=0,
		metavar='<int>', help='start of the analyzed sequence')
	parser.add_argument('--stop', required=False, type=int, default=42,
		metavar='<int>', help='end of the parsing ')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='end of the analyzed sequence')
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

	accs = []
	#splitting data into training and testing
	for train, test in seqio.cross_validation(seqs, arg.xvalid):

		#extracting trues and fakes out of the train data
		trues_train = [seq for label, seq in train if label == 1]
		fakes_train = [seq for label, seq in train if label == 0]

		#performing kmeans on training sets of trues and fakes
		#creating dataframes for trues and fakes
		trues_df = clust_lib.conv_data(trues_train)
		fakes_df = clust_lib.conv_data(fakes_train)

		#performing kmeans on training sets of trues and fakes
		trues_kmeans = KMeans(arg.k1).fit(trues_df)
		fakes_kmeans = KMeans(arg.k0).fit(fakes_df)

		#pca
		if arg.pca:
			clust_lib.pca_kmeans(trues_df, trues_train, arg.k1)
			clust_lib.pca_kmeans(fakes_df, fakes_train, arg.k0)

		#sorting sequences my created label
		trues_labeled = clust_lib.sorting(trues_train, trues_kmeans, arg.k1)
		fakes_labeled = clust_lib.sorting(fakes_train, fakes_kmeans, arg.k0)

		#creating pwm for each label
		trues_pwm = {}
		for t_label, t_seqs in trues_labeled.items():
			trues_pwm[t_label] = pwm.make_pwm(t_seqs)

		fakes_pwm = {}
		for f_label, f_seqs in fakes_labeled.items():
			fakes_pwm[f_label] = pwm.make_pwm(f_seqs)

		#extracting trues and fakes out of test data
		trues_test = [seq for label, seq in test if label == 1]
		fakes_test = [seq for label, seq in test if label == 0]

		#running the test set on the pwm and checking the accuracies
		tp, fn = clust_lib.acc(trues_test, trues_pwm, fakes_pwm)
		tn, fp = clust_lib.acc(fakes_test, fakes_pwm, trues_pwm)

		percentage = (tp+tn)/(tp+tn+fn+fp)

		accs.append(percentage)

	print('Accuracy:', f'{(sum(accs)/len(accs)):.4f}')
