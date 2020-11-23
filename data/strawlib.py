
import gzip
import math
import re
import sys
import random
import pandas as pd
from sklearn.cluster import KMeans
import pprint
import timeit
import os
import ast

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from kmodes.kmodes import KModes

def get_seqs(file, limit, start, end):
	seqs = []
	with gzip.open(file, mode='rt') as fp:
		lines = fp.read().splitlines()
		random.shuffle(lines)
		for i in range(limit):
			seqs.append(lines[i][start:end])
	#dup = set(seqs)
	#print('no duplicates:', len(dup), 'with duplicates:', len(seqs))

	return seqs

def dkl(p, q):
	d = 0
	for i in p:
		if p[i] != 0:
			d += p[i] * math.log2(p[i]/q[i])
	return d

def make_regex(trues):

	sim = {
		'A': {'A':0.97, 'C':0.01, 'G':0.01, 'T':0.01},
		'C': {'A':0.01, 'C':0.97, 'G':0.01, 'T':0.01},
		'G': {'A':0.01, 'C':0.01, 'G':0.97, 'T':0.01},
		'T': {'A':0.01, 'C':0.01, 'G':0.01, 'T':0.97},
		'R': {'A':0.49, 'C':0.01, 'G':0.49, 'T':0.01},
		'Y': {'A':0.01, 'C':0.49, 'G':0.01, 'T':0.49},
		'M': {'A':0.49, 'C':0.49, 'G':0.01, 'T':0.01},
		'K': {'A':0.01, 'C':0.01, 'G':0.49, 'T':0.49},
		'W': {'A':0.49, 'C':0.01, 'G':0.01, 'T':0.49},
		'S': {'A':0.01, 'C':0.49, 'G':0.49, 'T':0.01},
		'B': {'A':0.01, 'C':0.33, 'G':0.33, 'T':0.33},
		'D': {'A':0.33, 'C':0.01, 'G':0.33, 'T':0.33},
		'H': {'A':0.33, 'C':0.33, 'G':0.01, 'T':0.33},
		'V': {'A':0.33, 'C':0.33, 'G':0.33, 'T':0.01},
		'N': {'A':0.25, 'C':0.25, 'G':0.25, 'T':0.25},
	}

	reg = {
		'A': 'A',
		'C': 'C',
		'G': 'G',
		'T': 'T',
		'R': '[AG]',
		'Y': '[CT]',
		'M': '[AC]',
		'K': '[GT]',
		'W': '[AT]',
		'S': '[GC]',
		'B': '[CGT]',
		'D': '[AGT]',
		'H': '[ACT]',
		'V': '[ACG]',
		'N': '.',
	}

	# build regex from pwm
	pwm = make_pwm(trues)
	ntstr = ''
	restr = ''
	for i in range(len(pwm)):
		min_d = 1e6
		min_nt = None
		for nt in sim:
			d = dkl(pwm[i], sim[nt])
			if d < min_d:
				min_d = d
				min_nt = nt
		ntstr += min_nt
		restr += reg[min_nt]

	return restr, ntstr

def regex(trues, fakes, xv):

	sys.stderr.write('\nregex\n')
	TPR, TNR, PPV, NPV = 0, 0, 0, 0
	for x in range(xv):

		# collect testing and training sets
		ttrain, ttest, ftrain, ftest = [], [], [], []
		for i in range(len(trues)):
			if i % xv == x: ttest.append(trues[i])
			else:           ttrain.append(trues[i])
		for i in range(len(fakes)):
			if i % xv == x: ftest.append(fakes[i])
			else:           ftrain.append(fakes[i])

		# make regex
		restr, ntstr = make_regex(ttrain)

		# score against test set
		tp, tn, fp, fn = 0, 0, 0, 0
		for seq in ttest:
			match = re.search(restr, seq)
			#print(match)
			if match != None: tp += 1
			else:             fn += 1

		for seq in ftest:
			match = re.search(restr,seq)
			if match != None: fp += 1
			else:             tn += 1

		# gather performance stats
		tpr = tp / (tp + fn)
		tnr = tn / (tn + fp)
		ppv = tp / (tp + fp)
		npv = tn / (tn + fn)

		sys.stderr.write(f'set-{x} {ntstr} {tpr:.3f} {tnr:.3f} {ppv:.3f} {npv:.3f}\n')
		TPR += tpr
		TNR += tnr
		PPV += ppv
		NPV += npv

	return (2*TPR*PPV)/(TPR+PPV)

def make_pwm(seqs, boost=None):
	length = len(seqs[0])
	# create counts
	count = []
	for i in range(length): count.append({'A':0, 'C':0, 'G':0, 'T':0})
	total = 0
	for i in range(len(seqs)):
		seq = seqs[i]
		total += 1
		for i in range(len(seq)):
			if boost == None:
				count[i][seq[i]] += 1
			else:
				count[i][seq[i]] += boost[i]

	# create freqs
	freq = []
	for i in range(length): freq.append({})
	for i in range(length):
		for c in count[i]:
			freq[i][c] = count[i][c] / total

	return freq

def make_kmer(seqs, k, boost=None):
	length = len(seqs[0])
	count = {}
	total = 0
	for seq in seqs:
		for i in range(length - k):
			kmer = seq[i:i+k]
			if kmer not in count: count[kmer] = 1
			else:                 count[kmer] += 1
			total += 1

	if len(count) != 4 ** k: return None # some zero counts found

	freq = {}
	for kmer in count: freq[kmer] = count[kmer] / total
	return freq

def score_pwm(seq, pwm):
	p = 1
	for i in range(len(seq)):
		p *= pwm[i][seq[i]]
	return p

def score_kmer(seq, mm, k):
	p = 1
	for i in range(len(seq) - k):
		kmer = seq[i:i+k]
		p *= mm[kmer]
	return p

def kmer_threshold(trues, fakes, xv):

	sys.stderr.write('\nkmer_threshold\n')
	length = len(trues[0])
	k = 0
	kmers_full = True
	max_k_acc = 0
	max_k = None
	while kmers_full:
		k += 1
		sum_acc = 0

		for x in range(xv):

			# collect testing and training sets
			train, test = [], []
			for i in range(len(trues)):
				if i % xv == x: test.append(trues[i])
				else:           train.append(trues[i])

			# build mm
			km = make_kmer(train, k)
			if km == None:
				kmers_full = False
				break

			# find maximum thresholds
			vmax = 0
			for kmer in km:
				if km[kmer] > vmax: vmax = km[kmer]
			tmax = vmax ** length
			tmin = (0.25 ** (k)) ** length

			# find max t by stepping down tmax by half each time
			sys.stderr.write(f'set-{x} k-{k}')
			t = tmax
			n = 0
			acc_max = 0
			while True:
				t /= 2
				if t < tmin: break # no sense going below tmin

				# score against test set
				tp, tn, fp, fn = 0, 0, 0, 0
				for seq in test:
					s = score_kmer(seq, km, k)
					if s > t: tp += 1
					else:     fn += 1
				for i in range(len(test)): # fakes could be bigger, so limit
					seq = fakes[i]
					s = score_kmer(seq, km, k)
					if s > t: fp += 1
					else:     tn += 1

				if tp:
					tpr = tp / (tp + fn)
					ppv = tp / (tp + fp)
					acc = (tpr + ppv) / 2
					if acc > acc_max: acc_max = acc
			sys.stderr.write(f' {acc_max}\n')
			sum_acc += acc_max
			if sum_acc > max_k_acc:
				max_k_acc = sum_acc
				max_k = k

	return max_k_acc/xv

def pwm_evaluate(pwm, t, tsites, fsites):
	tp, tn, fp, fn = 0, 0, 0, 0

	for seq in tsites:
		s = score_pwm(seq, pwm)
		if s > t: tp += 1
		else:     fn += 1

	for seq in fsites:
		s = score_pwm(seq, pwm)
		if s > t: fp += 1
		else:     tn += 1

	return tp, tn, fp, fn

def pwm_threshold(trues, fakes, xv, x):

	#sys.stderr.write('\npwm_threshold\n')
	sum_acc = 0
	sum_acc_fake = 0
	#for x in range(xv):

	# collect testing and training sets
	train, test = [], []
	for i in range(len(trues)):
		if i % xv == x: test.append(trues[i])
		else:           train.append(trues[i])

	# build pwm
	pwm = make_pwm(train)

	# find maximum and minimum thresholds
	tmax = 1
	for i in range(len(pwm)):
		max = 0
		for c in pwm[i]:
			if pwm[i][c] > max: max = pwm[i][c]
		tmax *= max
	tmin = 0.25 ** len(pwm)

	# find max t by stepping down tmax by half each time
	#sys.stderr.write(f'set-{x} ')
	t = tmax
	n = 0
	acc_max = 0
	self_max = None


	while True:
		t /= 2
		if t < tmin: break # no sense going below tmin

		stp, stn, sfp, sfn = pwm_evaluate(pwm, t, train, fakes)

		# score against test set
		tp, tn, fp, fn = pwm_evaluate(pwm, t, test, fakes)
		if tp and stp:
			#tpr = (tp + tn) / (tp + fn + tn)
			#ppv = tp / (tp + fp)
			#acc = (1.25*tpr*ppv)/(tpr+ppv*0.25)
			acc = (tp + tn)/(tp+tn+fp+fn)

			#acc = tp/(tp+fn)
			#acc_fake = tn/(fp+tn)

			#acc_fake = tn/(fp+tn)
			if acc > acc_max:
				acc_max = acc
				#ssn = stp / (stp + sfn)
				#ssp = stp / (stp + sfp)
				#self_max = (1.25*ssn*ssp)/(ssn+ssp*0.25)

				self_max = (stp+stn)/(stp+stn+sfp+sfn)

				#self_max = (stp)/(stp+sfn)
				#self_max_fake = (sfn)/(sfp+stn) ###
	#sys.stderr.write(f' train:{self_max} test:{acc_max} t:{t}\n')
	sum_acc += acc_max

		#return t ###how to do this
		#sum_acc_fake += acc_fake
	#print(f'Fakes: {sum_acc_fake/xv:.4f}')

	return t
	#return (sum_acc / xv)

def pwm_vs_pwm(trues, fakes, xv):

	sys.stderr.write('\npwm_vs_pwm\n')
	#TPR, TNR, PPV, NPV, FSC = 0, 0, 0, 0, 0
	ACC = 0
	#false_p = []
	#false_n = []
	for x in range(xv):

		# collect testing and training sets
		ttrain, ttest, ftrain, ftest = [], [], [], []
		for i in range(len(trues)):
			if i % xv == x: ttest.append(trues[i])
			else:           ttrain.append(trues[i])
		for i in range(len(fakes)):
			if i % xv == x: ftest.append(fakes[i])
			else:           ftrain.append(fakes[i])

		# make pwms
		tpwm = make_pwm(ttrain)
		fpwm = make_pwm(ftrain)

		# score against test set
		tp, tn, fp, fn = 0, 0, 0, 0
		for seq in ttest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs:
				tp += 1
			else:
				fn += 1
				#false_n.append(seq)

		for seq in ftest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs:
				fp += 1
				#false_p.append(seq)
			else:
				tn += 1

		# gather performance stats
		#tpr = tp / (tp + fn)
		#tnr = tn / (tn + fp)
		#ppv = tp / (tp + fp)
		#npv = tn / (tn + fn)

		#sys.stderr.write(f'set-{x} {tpr:.3f} {tnr:.3f} {ppv:.3f} {npv:.3f}\n')
		#TPR += tpr
		#TNR += tnr
		#PPV += ppv
		#NPV += npv
		#FSC += (2*tpr*ppv)/(tpr+ppv)
		acc = (tp + tn)/(tp+tn+fp+fn)
		ACC += acc

	return ACC/xv

def boosted_pwms(trues, fakes, xv):

	sys.stderr.write('\nboosted_pwms\n')
	TPR, TNR, PPV, NPV, FSC = 0, 0, 0, 0, 0
	ACC = 0
	for x in range(xv):

		# collect testing and training sets
		ttrain, ttest, ftrain, ftest = [], [], [], []
		for i in range(len(trues)):
			if i % xv == x: ttest.append(trues[i])
			else:           ttrain.append(trues[i])
		for i in range(len(fakes)):
			if i % xv == x: ftest.append(fakes[i])
			else:           ftrain.append(fakes[i])

		# make pwms without boosting
		tpwm = make_pwm(ttrain)
		fpwm = make_pwm(ftrain)

		# score against test set and set boost
		tboost = []
		fboost = []
		tp, tn, fp, fn = 0, 0, 0, 0
		for seq in ttest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs:
				tp += 1
				tboost.append(1)
			else:
				fn += 1
				tboost.append(2)
		for seq in ftest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs:
				fp += 1
				fboost.append(2)
			else:
				tn += 1
				fboost.append(1)

		# re-make pwms with boosting
		tpwm = make_pwm(ttrain, boost=tboost)
		fpwm = make_pwm(ftrain, boost=fboost)

		# evaluate boosted PWMs
		tp, tn, fp, fn = 0, 0, 0, 0
		for seq in ttest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs: tp += 1
			else:       fn += 1
		for seq in ftest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs: fp += 1
			else:       tn += 1

		# gather performance stats
		acc = (tp + tn) / (tp + tn + fp + fn)
#		tpr = tp / (tp + fn)
#		tnr = tn / (tn + fp)
#		ppv = tp / (tp + fp)
#		npv = tn / (tn + fn)

#		sys.stderr.write(f'set-{x} {tpr:.3f} {tnr:.3f} {ppv:.3f} {npv:.3f}\n')
#		TPR += tpr
#		TNR += tnr
#		PPV += ppv
#		NPV += npv
#		FSC += (2*tpr*ppv)/(tpr+ppv)
		ACC += acc

	return acc / xv
import numpy as np
def kmeans(seqs, k, xv, x):
	train, test = [], []
	'''
	for i in range(len(seqs)):
		train.append(seqs[i])

	'''

	for i in range(len(seqs)):
		if i % xv == x:
			test.append(seqs[i])
		else:
			train.append(seqs[i])

	assert(k<=len(train))

	list_bases = {'A': 1.0, 'C': 2.0, 'G': 3.0, 'T': 4.0}

	converted_sequences = []
	for i in range(len(train)):
		converted_sequences.append([])


	for item in range(len(train)):
		for i in train[item]:
			if i in list_bases.keys():
				converted_sequences[item].append(list_bases[i])


	df = pd.DataFrame(converted_sequences)

	headers = []
	for i in range(len(converted_sequences[0])):
		headers.append(str(f'p{i}'))
	df.columns = headers

	#kmeans = KMeans(k).fit_predict(df)
	#.fit(df) = initialization values
	#fit_predict(df) = labeling results after running model on data

	kmeans = KMeans(k).fit(df)
	#centroids = kmeans.cluster_centers_


	seqs_by_label = {}
	for label, seq in zip(kmeans.labels_, train):
		###
		if label in seqs_by_label:
			seqs_by_label[label].append(seq)
		else:
			seqs_by_label[label] = []
			seqs_by_label[label].append(seq)
	assert(len(seqs_by_label.keys()) == k)

	PCA_kmeans(df, seqs, k)
	return test, seqs_by_label

def PCA_kmeans(df, seqs, k):
	x = df.values
	x_std = StandardScaler().fit_transform(df)

	#f, ax = plt.subplots(figuresize = ())

	#eigenvectors and eigenvalues
	mean_vec = np.mean(x_std, axis = 0)
	cov_mat = np.cov(x_std.T)
	eig_vals, eig_vecs = np.linalg.eig(cov_mat)

	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	eig_pairs.sort(key= lambda x: x[0], reverse = True)
	tot = sum(eig_vals)
	var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse = True)]
	cum_var_exp = np.cumsum(var_exp)

	plt.figure(figsize = (10,5))
	plt.bar(range(len(seqs[0])), var_exp, alpha = 0.3333, align = 'center', label = 'individual explained variance', color = 'gold')
	#plt.step(range(len(seqs[0])), cum_var_exp, where='mid', label = 'cumulative explained variance')
	plt.ylabel('explained_variance_ratio')
	plt.xlabel('Principal Components')
	plt.legend(loc='best')
	#print('general1')
	plt.show()

	#PCA Analysis
	pca = PCA(n_components = len(seqs[0]))
	x_9d = pca.fit_transform(x_std)
	plt.scatter(x_9d[:,0], x_9d[:,1], c ='goldenrod', alpha = 0.5)
	#print('general2')
	plt.show()

	#visualization with k-means clustering
	kmeans = KMeans(k)
	label_color_map = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c', 5: 'm'}
	x_clustered = kmeans.fit_predict(x_9d)
	label_color = [label_color_map[i] for i in x_clustered]

	plt.scatter(x_9d[:,0], x_9d[:,1], c = label_color, alpha = 0.5)
	#print('general3')
	plt.show()


def optimal_length(trues, fakes, limit_trues, limit_fakes, kt, kf, xv):
	if 'don' in trues:
		beg = 23
		end = 26
		result_beg = 23
		result_end = 26
		highest_acc = 0.0

		for j in range(beg+1):
			for i in range(43-end):
				true_seqs = get_seqs(trues, limit_trues, beg, end)
				fake_seqs = get_seqs(fakes, limit_fakes, beg, end)

				acc = kmeans_pwm(true_seqs, fake_seqs, kt, kf, xv)
				print(beg, end)
				if acc > highest_acc:
					highest_acc = acc
					result_end = end
					result_beg = beg
				end += 1
			end = 25
			beg -= 1

		return result_beg, result_end, highest_acc


	elif 'acc' in trues:
		beg = 20
		end = 23
		highest_acc = 0.0
		result_beg = 20
		result_end = 22

		for i in range(43-end):
			for j in range(beg+1):
				print(beg, end)
				true_seqs = get_seqs(trues, limit_trues, beg, end)
				fake_seqs = get_seqs(fakes, limit_fakes, beg, end)

				acc = kmeans_pwm(true_seqs, fake_seqs, kt, kf, xv)
				if acc > highest_acc:
					highest_acc = acc
					result_beg = beg
					result_end = end
				beg -= 1
			beg = 20
			end += 1

		return result_beg, result_end, highest_acc

def manhattan_distance(matrix, start):
	storing_manhattan_distance = {}
	for label, position_matrix in matrix.items():
		opening_dictionary = {}
		for position in range(len(position_matrix)):
			opening_dictionary[position] = position_matrix[position]

		for comparing_label, comparing_matrix in matrix.items():
			if comparing_label != label:
				if ('{} - {}'.format(comparing_label, label)) not in storing_manhattan_distance:
					storing_manhattan_distance['{} - {}'.format(label, comparing_label)] = 0.0
					for position in range(len(comparing_matrix)):
						#storing_manhattan_distance['{} - {}'.format(label, comparing_label)].append('p{}'.format(position))
						#storing_manhattan_distance['{} - {}'.format(label, comparing_label)][position+start] = {}
						for base, probability in comparing_matrix[position].items():
							if base in opening_dictionary[position]:
								storing_manhattan_distance['{} - {}'.format(label, comparing_label)] += float(f'{abs(opening_dictionary[position][base]-comparing_matrix[position][base]):.4f}')
								#storing_manhattan_distance['{} - {}'.format(label, comparing_label)][position+start][base] = (f'{abs(opening_dictionary[position][base]-comparing_matrix[position][base]):.4f}')
								#storing_manhattan_distance['{} - {}'.format(label,comparing_label)].append(f'{abs(opening_dictionary[position][base]-comparing_matrix[position][base]):.4f}')

	return(storing_manhattan_distance)

def kmeans_pwm(trues, fakes, kt, kf, xv, start):
	storing_manhattan_distance = {}
	accuracy = []
	false_p = []
	false_n = []
	for x in range(xv):
		print(f'iteration {x}')

		true_test, true_seqs_by_label = kmeans(trues, kt, xv, x)

		t_pwm = {}
		for t_label, t_seqs in true_seqs_by_label.items():
			t_pwm[t_label] = make_pwm(t_seqs)

		#true_computing_manh = manhattan_distance(t_pwm, start)
		#storing_manhattan_distance['true {}'.format(x)] = true_computing_manh
		#print('trues')
		#pprint.pprint(true_computing_manh)

		fake_test, fake_seqs_by_label= kmeans(fakes, kf, xv, x)

		f_pwm = {}
		for f_label, f_seqs in fake_seqs_by_label.items():
			f_pwm[f_label] = make_pwm(f_seqs)

		#fake_computing_manh = manhattan_distance(f_pwm, start)
		#storing_manhattan_distance['fake {}'.format(x)] = fake_computing_manh
		#print('fakes')
		#pprint.pprint(fake_computing_manh)

		tp = 0
		fn = 0
		tn = 0
		fp = 0

		for true_test_seq in true_test:
			t_final_score = 0.0
			for t_label, t_matrix in t_pwm.items():
				t_score = 1
				for t_base, t_probability_of_base in zip(true_test_seq, t_matrix):
					t_score *= t_probability_of_base[t_base]
				if t_score > t_final_score:
					t_final_score = t_score
			f_final_score = 0.0
			for f_label, f_matrix in f_pwm.items():
				f_score = 1
				for f_base, f_probability_of_base in zip(true_test_seq, f_matrix):
					f_score *= f_probability_of_base[f_base]
				if f_score > f_final_score:
					f_final_score = f_score
			if t_final_score > f_final_score:
				tp += 1
			elif t_final_score < f_final_score:
				fn += 1
				false_n.append(true_test_seq)


		for fake_test_seq in fake_test:

			t_final_score = 0.0
			for t_label, t_matrix in t_pwm.items():
				t_score = 1
				for t_base, t_probability_of_base in zip(fake_test_seq, t_matrix):
					t_score *= t_probability_of_base[t_base]
				if t_score > t_final_score:
					t_final_score = t_score
				#print(t_score)

			f_final_score = 0.0
			for f_label, f_matrix in f_pwm.items():
				f_score = 1
				for f_base, f_probability_of_base in zip(fake_test_seq, f_matrix):
					f_score *= f_probability_of_base[f_base]
				if f_score > f_final_score:
					f_final_score = f_score
				#print(f_score)


			if f_final_score > t_final_score:
				tn += 1
			elif t_final_score > f_final_score:
				fp += 1
				false_p.append(fake_test_seq)

		acc = (tp+tn)/(tp+tn+fn+fp)


		accuracy.append(acc)




			# do the clustering
			# create k PWMs from k clusters
				# find optimal threshold for each PWM
				# score
			# aggregate scores for these PWMs
		# aggregate scores for this k
	# report highest performance
	#print('Accuracy:', '{:.4f}'.format(sum(accuracy)/len(accuracy)))
	#pprint.pprint(storing_manhattan_distance)
	return sum(accuracy)/len(accuracy)

def exhaustive_method(trues, fakes, xv, beg, end):
	storing_results = {}
	for x in range(xv):
		print(f'cross_val iteration {x}')

		true_train, true_test = [], []
		for i in range(len(trues)):
			if i % xv == x:
				true_test.append(trues[i])
			else:
				true_train.append(trues[i])

		fake_train, fake_test = [], []
		for i in range(len(fakes)):
			if i % xv == x:
				fake_test.append(fakes[i])
			else:
				fake_train.append(fakes[i])

		#f_pwm = make_pwm(fake_train)


		accuracy = 0.0
		#beg = 16
		#end = 24
		NT = ['A', 'C', 'G', 'T']
		for i in range(beg, end, 1):
			for nt in NT:
				name_set1 = f'set-{i}{nt}+'
				name_set2 = f'set-{i}{nt}-'
				set1, set2 = splitter(true_train, i, nt)

				fake_name_set1 = f'fake_set-{i}{nt}+'
				fake_name_set2 = f'fake_set-{i}{nt}-'

				fake_set1, fake_set2 = splitter(fake_train, i, nt)

				if len(set1) != 0 and len(set2) != 0 and len(fake_set1) != 0 and len(fake_set2) != 0:
					t1_pwm = make_pwm(set1)
					t2_pwm = make_pwm(set2)

					f1_pwm = make_pwm(fake_set1)
					f2_pwm = make_pwm(fake_set2)

					tp = 0
					fn = 0
					tn = 0
					fp = 0

					for true_test_seq in true_test:

						t_final_score = 0
						t1_score = 1
						for t_base, t_probability_of_base in zip(true_test_seq, t1_pwm):
							t1_score *= t_probability_of_base[t_base]

						t2_score = 1
						for t2_base, t2_probability_of_base in zip(true_test_seq, t2_pwm):
							t2_score *= t2_probability_of_base[t2_base]

						if t1_score > t2_score:
							t_final_score = t1_score
						elif t1_score < t2_score:
							t_final_score = t2_score


						f_final_score = 0
						f1_score = 1
						for f1_base, f1_probability_of_base in zip(true_test_seq, f1_pwm):
							f1_score *= f1_probability_of_base[f1_base]

						f2_score = 1
						for f2_base, f2_probability_of_base in zip(true_test_seq, f2_pwm):
							f2_score *= f2_probability_of_base[f2_base]



						if f1_score > f2_score:
							f_final_score = f1_score
						elif f1_score < f2_score:
							f_final_score = f2_score


						if t_final_score > f_final_score:
							tp += 1
						elif t_final_score < f_final_score:
							fn += 1

					for fake_test_seq in fake_test:
						t_final_score = 0
						t1_score = 1
						for t_base, t_probability_of_base in zip(fake_test_seq, t1_pwm):
							t1_score *= t_probability_of_base[t_base]

						t2_score = 1
						for t2_base, t2_probability_of_base in zip(fake_test_seq, t2_pwm):
							t2_score *= t2_probability_of_base[t2_base]

						if t1_score > t2_score:
							t_final_score = t1_score
						elif t1_score < t2_score:
							t_final_score = t2_score


						f_final_score = 0

						f1_score = 1
						for f1_base, f1_probability_of_base in zip(fake_test_seq, f1_pwm):
							f1_score *= f1_probability_of_base[f1_base]

						f2_score = 1
						for f2_base, f2_probability_of_base in zip(fake_test_seq, f2_pwm):
							f2_score *= f2_probability_of_base[f2_base]

						if f1_score > f2_score:
							f_final_score = f1_score
						elif f1_score < f2_score:
							f_final_score = f2_score



						if f_final_score > t_final_score:
							tn += 1
						elif t_final_score > f_final_score:
							fp += 1
					acc = (tp+tn)/(tp+tn+fn+fp)
					files = ['true_set1', 'true_set2','fake_set1', 'fake_set2']
					key = '{}, {}, {}, {}'.format(name_set1, name_set2, fake_name_set1, fake_name_set2)
					if key not in storing_results:
						storing_results[key] = [[],[],[],[],[]]
						storing_results[key][0].append(float(f'{acc:.4f}'))
						storing_results[key][1].append(len(set1))
						storing_results[key][2].append(len(set2))
						storing_results[key][3].append(len(fake_set1))
						storing_results[key][4].append(len(fake_set2))
					else:
						storing_results[key][0].append(float(f'{acc:.4f}'))
						storing_results[key][1].append(len(set1))
						storing_results[key][2].append(len(set2))
						storing_results[key][3].append(len(fake_set1))
						storing_results[key][4].append(len(fake_set2))
	#pprint.pprint(storing_results)
	for file, values in storing_results.items():
		for i in range(len(values)):
			if i == 0:
				storing_results[file][i] = (f'{sum(values[i])/len(values[i]):.4f}')
			else:
				storing_results[file][i] = (f'{sum(values[i])/len(values[i])}')
	#pprint.pprint(storing_results)
	return storing_results

def splitter(trues, i, nt):
	set1 = []
	set2 = []
	for seq in trues:
		if seq[i] == nt:
			set1.append(seq)
		else:
			set2.append(seq)
	return set1, set2

def train_and_test_in_kmeans(trues, fakes, kt, xv, nt, nf):

	accuracy = {}
	for x in range(xv):
		print(f'iteration {x}')

		true_train, true_test = [], []
		for i in range(len(trues)):
			if i % xv == x:
				true_test.append(trues[i])
			else:
				true_train.append(trues[i])

		fake_train, fake_test = [], []
		for i in range(len(fakes)):
			if i % xv == x:
				fake_test.append(fakes[i])
			else:
				fake_train.append(fakes[i])

		train = true_train + fake_train
		test = true_test + fake_test

		assert(kt<=len(train))

		list_bases = {'A': 1.0, 'C': 2.0, 'G': 3.0, 'T': 4.0}

		train_converted_sequences = []
		for i in range(len(train)):
			train_converted_sequences.append([])

		for item in range(len(train)):
			for i in train[item]:
				if i in list_bases.keys():
					train_converted_sequences[item].append(list_bases[i])


		df = pd.DataFrame(train_converted_sequences)

		headers = []
		for i in range(len(train_converted_sequences[0])):
			headers.append(str(f'p{i}'))
		df.columns = headers

		clustering = KMeans(kt).fit(df)
		train_labels = clustering.labels_

		#################################
		pca_try_clustering = KMeans(kt).fit_predict(df)
		std_labels = StandardScaler().fit_transform(df)

		pca = PCA(n_components=len(trues[0]))
		principalComponents = pca.fit_transform(std_labels)

		PC = range(1, pca.n_components_+1)
		plt.bar(PC, pca.explained_variance_ratio_, color='gold')
		plt.xlabel('Principal Components')
		plt.ylabel('Variance %')
		plt.xticks(PC)
		PCA_components = pd.DataFrame(principalComponents)
		#prins out the ranks with the highest variance
		plt.show()

		plt.scatter(PCA_components[0], PCA_components[1], alpha=.3, color='gold')
		plt.xlabel('PCA 1')
		plt.ylabel('PCA 2')
		plt.show()

		model = KMeans(kt)
		model.fit(PCA_components.iloc[:,:2])
		labels = model.predict(PCA_components.iloc[:,:2])
		plt.scatter(PCA_components[0], PCA_components[1], c=labels)
		plt.show()


		#################################

		test_converted_sequences = []
		for i in range(len(test)):
			test_converted_sequences.append([])

		for item in range(len(test)):
			for i in test[item]:
				if i in list_bases.keys():
					test_converted_sequences[item].append(list_bases[i])

		test_df = pd.DataFrame(test_converted_sequences)


		test_labels = clustering.predict(test_df)

		dictionary = {}
		true = 'true'
		fake = 'fake'
		#dictionary['true'] = {}
		#dictionary['fake'] = {}


		for label in test_labels:
			dictionary[label] = {}

		for label, seq in zip(test_labels[:len(true_test)], test[:len(true_test)]):
			if true in dictionary[label]:
				dictionary[label]['true'] += 1
			else:
				dictionary[label]['true'] = 1
		for label, seq in zip (test_labels[len(true_test):], test[len(true_test):]):
			if fake in dictionary[label]:
				dictionary[label]['fake'] += 1
			else:
				dictionary[label]['fake'] = 1
		print(dictionary)
		for label, info in dictionary.items():
			print('Label:', label)
			for tf, number in info.items():
				print(tf, f'{number/len(true_test):.4f}')
		'''
		for label, seq in zip (test_labels[:len(true_test)], test[:len(true_test)]):
			if label in dictionary['true']:
				dictionary['true'][label].append(seq)
			else:
				dictionary['true'][label] = []
				dictionary['true'][label].append(seq)
		for label, seq in zip (test_labels[len(true_test):], test[len(true_test):]):
			if label in dictionary['fake']:
				dictionary['fake'][label].append(seq)
			else:
				dictionary['fake'][label] = []
				dictionary['fake'][label].append(seq)
		for true_fake, labels_seqs in dictionary.items():
			for label, list_of_seqs in labels_seqs.items():
				print(true_fake, label, len(list_of_seqs),f'{len(list_of_seqs)/len(true_test):.4f}')

		'''

		#print(m, i, dictionary)
	return

#from apyori import apriori
def using_apriori(trues, fakes, xv, nt, nf):
	plt.style.use('seaborn-deep')

	accuracy = []
	for x in range(xv):
		print(f'iteration {x}')

		true_train, true_test = [], []
		for i in range(len(trues)):
			if i % xv == x:
				true_test.append(trues[i])
			else:
				true_train.append(trues[i])

		fake_train, fake_test = [], []
		for i in range(len(fakes)):
			if i % xv == x:
				fake_test.append(fakes[i])
			else:
				fake_train.append(fakes[i])

		#train = true_train + fake_train
		#test = true_test + fake_test

		#list_bases = {'A': 1.0, 'C': 2.0, 'G': 3.0, 'T': 4.0}

		true_train_converted_sequences = []
		for i in range(len(true_train)):
			true_train_converted_sequences.append([])

		for item in range(len(true_train)):
			for i in range(len(true_train[item])):
				true_train_converted_sequences[item].append('{}@{}'.format(true_train[item][i], i))
				#if i in list_bases.keys():
					#train_converted_sequences[item].append(list_bases[i])
		fake_train_converted_seqs = []
		for i in range(len(fake_train)):
			fake_train_converted_seqs.append([])

		for fitem in range(len(fake_train)):
			for f in range(len(fake_train[fitem])):
				fake_train_converted_seqs[fitem].append('{}@{}'.format(fake_train[fitem][f], f))

		true_rules = apriori(true_train_converted_sequences, min_support = 0.80)
		true_results = list(true_rules)
		#for i in true_results:
			#print()
			#print(i, type(i))
		#print(len(true_results))

		support_values = [item[1] for item in true_results]
		listRules = [list(true_results[i][0]) for i in range(0,len(true_results))]
		#print('\n','\n','\n','\n','\n')
		#print(support_values)
		#print(listRules)
		true_apriori_values = {}
		for rules, values in zip(listRules, support_values):
			true_apriori_values[', '.join(rules)] = values
		#pprint.pprint(true_apriori_values)
		#sorted_x = sorted(true_apriori_values.items(), reverse = True, key=lambda x: x[1])
		#print(sorted_x)
		#print('\n','\n','\n','\n','\n')

		fake_rules = apriori(fake_train_converted_seqs, min_support = 0.40)
		fake_results = list(fake_rules)
		#for i in fake_results:
			#print()
			#print(i)
		#print(len(fake_results))
		fake_support_values = [item[1] for item in fake_results]
		fakelistRules = [list(fake_results[i][0]) for i in range (0, len(fake_results))]

		fake_apriori_values = {}
		for fake_rules, fake_values in zip(fakelistRules, fake_support_values):
			fake_apriori_values[', '.join(fake_rules)] = fake_values
		#pprint.pprint(fake_apriori_values)


		tp = 0
		fn = 0

		for true_test_seq in true_test:
			temp_list = []
			for bases, value in true_apriori_values.items():
				#print(bases, values, len(bases))
				if len(bases) >= 16 and value < 1.0:
					#print(bases, values)
					list_of_bases = bases.split(', ')
					temp_dict = {}
					for base in list_of_bases:
						x = base.split('@')
						nuc, loc = x[0], int(x[1])
						temp_dict[loc] = nuc
					temp_list.append((all(true_test_seq[loc] == nuc for loc, nuc in temp_dict.items())))
			#print(temp_list, true_test_seq)
			if True in temp_list:
				tp += 1
			else:
				fn += 1
		print('tp:',tp, 'fn:', fn)

		tn = 0
		fp = 0
		for fake_test_seq in fake_test:
			f_temp_list = []
			for f_bases, f_value in true_apriori_values.items():
				if len(f_bases) >= 22 and value < 1.0:
					#print(f_bases, f_value)
					list_of_f_bases = f_bases.split(', ')
					f_temp_dict = {}
					for f_base in list_of_f_bases:
						x = f_base.split('@')
						f_nuc, f_loc = x[0], int(x[1])
						f_temp_dict[f_loc] = f_nuc
					f_temp_list.append((all(fake_test_seq[f_loc] == f_nuc for f_loc, f_nuc in f_temp_dict.items())))
			#print(f_temp_list, fake_test_seq)
			if True in f_temp_list:
				fp += 1
			else:
				tn += 1
		print('tn:', tn, 'fp:', fp)
		#print(len(true_test), len(fake_test))
		acc = (tp+tn)/(tp+tn+fn+fp)
		print(acc)
		accuracy.append(acc)
		#pprint.pprint(true_apriori_values)
	return sum(accuracy)/ len(accuracy)
		#sys.exit()
		#print(len(bases), bases, list_of_bases, value)

