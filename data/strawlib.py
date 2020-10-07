
import gzip
import math
import re
import sys
import random
import pandas as pd
from sklearn.cluster import KMeans
import pprint
import timeit

def get_seqs(file, limit, start, end):
	seqs = []
	with gzip.open(file, mode='rt') as fp:
		lines = fp.read().splitlines()
		random.shuffle(lines)
		for i in range(limit):
			seqs.append(lines[i][start:end])

	dup = set(seqs)
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
			if ts > fs: tp += 1
			else:       fn += 1

		for seq in ftest:
			ts = score_pwm(seq, tpwm)
			fs = score_pwm(seq, fpwm)
			if ts > fs: fp += 1
			else:       tn += 1

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

def kmeans(seqs, k, xv, x):
	train, test = [], []

	for i in range(len(seqs)):
		train.append(seqs[i])


	'''
	for i in range(len(seqs)):
		if i % xv == x:
			test.append(seqs[i])
		else:
			train.append(seqs[i])
	'''
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

	kmeans = KMeans(k).fit(df)
	centroids = kmeans.cluster_centers_

	seqs_by_label = {}
	for label, seq in zip(kmeans.labels_, train):
		###
		if label in seqs_by_label:
			seqs_by_label[label].append(seq)
		else:
			seqs_by_label[label] = []
			seqs_by_label[label].append(seq)

	assert(len(seqs_by_label.keys()) == k)

	return train, seqs_by_label

def threshold_for_kpwm(seqs_by_label, xv, x):
	storing_thr = {}

	for true_thr_label in seqs_by_label:
		trues_thr = seqs_by_label[true_thr_label]
		fakes_thr = []

		for thr_label, thr_seqs in seqs_by_label.items():
			if thr_label != true_thr_label:
				fakes_thr += thr_seqs

		storing_thr[true_thr_label] = pwm_threshold(trues_thr, fakes_thr, xv, x)

	return storing_thr


	#print(storing_thr)

def optimal_length(trues, fakes, limit_trues, limit_fakes, kt, kf, xv):
	if 'don' in trues:
		beg = 23
		end = 27
		result_beg = 23
		result_end = 27
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
			end = 27
			beg -= 1

		return result_beg, result_end, highest_acc


	elif 'acc' in trues:
		beg = 20
		end = 24
		highest_acc = 0.0
		result_beg = 20
		result_end = 24

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

def manhattan_distance():
	pass

def kmeans_pwm(trues, fakes, kt, kf, xv):
	start = timeit.default_timer()
	#sys.stderr.write('\npkmeans_pwm\n')
	accuracy = []
	for x in range(xv):
		print(f'iteration {x}')

		true_test, true_seqs_by_label = kmeans(trues, kt, xv, x)

		t_pwm = {}
		for t_label, t_seqs in true_seqs_by_label.items():
			t_pwm[t_label] = make_pwm(t_seqs)


		fake_test, fake_seqs_by_label= kmeans(fakes, kf, xv, x)

		f_pwm = {}
		for f_label, f_seqs in fake_seqs_by_label.items():
			f_pwm[f_label] = make_pwm(f_seqs)

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

	#stop = timeit.default_timer()
	#execution_time = stop - start

	#print("Program Executed in "+str(execution_time))
	return sum(accuracy)/len(accuracy)

