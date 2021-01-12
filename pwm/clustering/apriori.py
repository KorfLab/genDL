import argparse
import sys
import random
from gendl import seqio, pwm
from pwm.clustering import clust_lib
from apyori import apriori

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Perform Kmeans Clustering on genetic sequence')
	parser.add_argument('--file1', required=True, type=str,
		metavar='<file>', help='fasta file1')
	parser.add_argument('--file0', required=True, type=str,
		metavar='<file>', help='fasta file0')
	parser.add_argument('--xvalid', required=False, type=int, default=4,
		metavar='<int>', help='x-fold cross-validation [%(default)s]')
	parser.add_argument('--mins1', required=False, type=int, default=0.2,
		metavar='<int>', help='minimum support parameter for file 1')
	parser.add_argument('--mincon1', required=False, type=int, default=0.7,
		metavar='<int>', help='minimum confidence parameter for file 1')
	parser.add_argument('--mins0', required=False, type=int, default=0.2,
		metavar='<int>', help='minimum support parameter for file 0')
	parser.add_argument('--mincon0', required=False, type=int, default=0.5,
		metavar='<int>', help='minimum confidence parameter for file 0')
	parser.add_argument('--start', required=False, type=int, default=0,
		metavar='<int>', help='start of the analyzed sequence')
	parser.add_argument('--stop', required=False, type=int, default=42,
		metavar='<int>', help='end of the parsing ')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='end of the analyzed sequence')

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
		#each sequence is converted to the list of lists, where each base is separated and also has a position available

		#performing necessary conversion prior apriori
		trues_train_apriori = clust_lib.list_position(trues_train, arg.start, arg.stop)
		fakes_train_apriori = clust_lib.list_position(fakes_train, arg.start, arg.stop)

		#apriori on converted train sets and getting a set of rules
		trues_rules = clust_lib.appr(trues_train_apriori, arg.mins1, arg.mincon1)
		fakes_rules = clust_lib.appr(fakes_train_apriori, arg.mins0, arg.mincon0)

		#extracting trues and fakes out of test data
		trues_test = [seq for label, seq in test if label == 1]
		fakes_test = [seq for label, seq in test if label == 0]

		#kalmagorov-smirnov test = sanity test check? = can use for the rest of clusterings
		#testing the sequences against those rules
		#see the accuracy (as I did it)












