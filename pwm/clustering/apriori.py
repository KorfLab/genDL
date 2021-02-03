import argparse
import sys
import random
from gendl import seqio, pwm
from pwm.clustering import clust_lib
from apyori import apriori
from string import ascii_letters

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
	parser.add_argument('--real', required=False, action='store_true',
		help='If the data is real "possibly remove it later"')
	arg = parser.parse_args()

	#probability should be <= 1.0
	assert(arg.mins1 <= 1.0 and arg.mins1 >= 0.0)
	assert(arg.mins1 <= 1.0 and arg.mins1 >= 0.0)

	if arg.seed:
		random.seed(arg.seed)

	#read sequences and create a dataframe out of them
	seqs1 = [(1, seq[arg.start:arg.stop]) for name, seq in seqio.read_fasta(arg.file1)]
	seqs0 = [(0, seq[arg.start:arg.stop]) for name, seq in seqio.read_fasta(arg.file0)]
	seqs = seqs1 + seqs0
	random.shuffle(seqs)

	def acc_appr(train_set, rule):
		match, nmatch = clust_lib.preping_for_pwm(train_set, rule)
		mpwm = pwm.make_pwm(match)
		npwm = pwm.make_pwm(nmatch)

		tp, tn, fp, fn = 0, 0, 0, 0

		for entry in test:
			label, seq = entry

			mscore = pwm.score_pwm(mpwm, seq)
			nscore = pwm.score_pwm(npwm, seq)

			if label == 1:
				if  mscore > nscore:
					tp += 1
				else:
					fn += 1
			elif label == 0:
				if nscore > mscore:
					tn += 1
				else:
					fp += 1
		acc = (tp+tn)/(tp+tn+fp+fn)
		return (acc)



	accs = {}
	#splitting data into training and testing
	for train, test in seqio.cross_validation(seqs, arg.xvalid):
		#extracting trues and fakes out of the train data
		trues_train = [seq for label, seq in train if label == 1]
		fakes_train = [seq for label, seq in train if label == 0]

		#apriori on converted train sets and getting a set of rules
		trues_rules = clust_lib.appr(trues_train, arg.start, arg.stop, arg.mins1)
		fakes_rules = clust_lib.appr(fakes_train, arg.start, arg.stop, arg.mins0)

		#returns rules if asked
		if arg.rules:
			print('Rules for file1')
			for i in trues_rules:
				print('Association rule:', i[0],'Support:', i[1])
			print('\n')
			print('Rules for file0')
			for i in fakes_rules:
				print('Association rule:', i[0], 'Support:', i[1])
			print('\n')
			break

		#filtering out rules present in both datasets
		trues_rules, fakes_rules = clust_lib.filtering(trues_rules, fakes_rules)
		assert(len(trues_rules) > 0)

		#extracting trues and fakes out of test data
		trues_test = [seq for label, seq in test if label == 1]
		fakes_test = [seq for label, seq in test if label == 0]

		#preparing training set to later use it for pwm
		train_for_pwm = [seq for label, seq in train]
		train_for_pwm = clust_lib.list_position(train_for_pwm, arg.start, arg.stop)

		#creating pwm from the training set and apriori rules

		for rule in trues_rules:
			acc = clust_lib.acc_appr(train_for_pwm, rule, test)

			if rule[0] not in accs:
				accs[rule[0]] = [] #['file1']
			accs[rule[0]].append(acc)

		if arg.real:
			for rule in fakes_rules:
				f_acc = clust_lib.acc_appr(train_for_pwm, rule, test)

				if rule[0] not in accs:
					accs[rule[0]] = [] #['file0']
				accs[rule[0]].append(f_acc)

	if not arg.rules:
		avg_accs = {}
		for rule, value in accs.items():
			avg_accs[rule] = sum(value)/len(value)
		avg_accs = dict(sorted(avg_accs.items(), key=lambda item: item[1]))

		for key, value in avg_accs.items():
			print(key, value)





