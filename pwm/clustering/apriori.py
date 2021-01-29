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
	parser.add_argument('--mins1', required=False, type=float, default=0.8,
		metavar='<float>', help='minimum support parameter for file 1, default = 0.8')
	parser.add_argument('--mins0', required=False, type=float, default=0.8,
		metavar='<float>', help='minimum support parameter for file 0, default = 0.8')
	parser.add_argument('--start', required=False, type=int, default=0,
		metavar='<int>', help='start of the analyzed sequence')
	parser.add_argument('--stop', required=False, type=int, default=42,
		metavar='<int>', help='end of the parsing ')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='end of the analyzed sequence')
	parser.add_argument('--rules', required=False, action='store_true',
		help='returns association rules for files')
	arg = parser.parse_args()

	#probability should be <= 1.0
	assert(arg.mins1 <= 1.0 and arg.mins1 >= 0.0)
	assert(arg.mins1 <= 1.0 and arg.mins1 >= 0.0)

	if arg.seed:
		random.seed(arg.seed)

	'''
	seqs1 = [(1, seq[arg.start:arg.stop]) for seq in seqio.read_raw(arg.file1)]
	seqs0 = [(0, seq[arg.start:arg.stop]) for seq in seqio.read_raw(arg.file0)]

	'''
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

		#apriori on converted train sets and getting a set of rules
		trues_rules = clust_lib.appr(trues_train, arg.start, arg.stop, arg.mins1)
		fakes_rules = clust_lib.appr(fakes_train, arg.start, arg.stop, arg.mins0)

		#getting rid of rules present in both subsets
		clust_lib.appr_check(trues_rules, fakes_rules)

		#returns rules
		if arg.rules:
			print('Rules for file1')
			for i in trues_rules:
				print('Association rule:', i[0],'Support:', i[1])
			print('\n')
			print('Rules for file0')
			for i in fakes_rules:
				print('Association rule:', i[0], 'Support:', i[1])
			print('\n')
		sys.exit()
		#extracting trues and fakes out of test data
		trues_test = [seq for label, seq in test if label == 1]
		fakes_test = [seq for label, seq in test if label == 0]

		#converting test set to the same format used by apriori
		trues_test = clust_lib.list_position(trues_test, arg.start, arg.stop)
		#all possible association rules
		#pwm (relatable to pwm)
		#run
		fakes_test = clust_lib.list_position(fakes_test, arg.start, arg.stop)

		#calculating accuracies
		#print('file1')
		#tp, fn = clust_lib.apr_acc(trues_test, trues_rules, fakes_rules)
		#print('yaaaa', tp, fn, len(trues_test), tp/(len(trues_test)))
		print('file0')
		tn, fp = clust_lib.apr_acc(fakes_test, fakes_rules, trues_rules)
		print(tn, fp, len(fakes_test), 'ahahah', tn/len(fakes_test))

		#percentage = (tp+tn)/(tp+tn+fn+fp)

		#accs.append(percentage)

	#print('Accuracy:', f'{(sum(accs)/len(accs)):.4f}')












