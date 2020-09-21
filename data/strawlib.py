
import gzip
import math
import re
import sys
import random
import pandas as pd
from sklearn.cluster import KMeans
import pprint

def get_seqs(file, limit):
	seqs = []
	with gzip.open(file, mode='rt') as fp:
		lines = fp.read().splitlines()
		random.shuffle(lines)
		for i in range(limit):
			seqs.append(lines[i])

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

def pwm_threshold(trues, fakes, xv):

	sys.stderr.write('\npwm_threshold\n')
	sum_acc = 0
	sum_acc_fake = 0
	for x in range(xv):

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
		sys.stderr.write(f'set-{x} ')
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
		sys.stderr.write(f' train:{self_max} test:{acc_max} t:{t}\n')
		sum_acc += acc_max

		return t ###how to do this
		#sum_acc_fake += acc_fake
	#print(f'Fakes: {sum_acc_fake/xv:.4f}')

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

def kmeans_pwm(trues, fakes, k, xv):
	#print(trues)
	#print(fakes)
	#sys.stderr.write('\npkmeans_pwm\n')
	pwm_thr_it = {}
	pwm_for_true_label_it = {}
	for x in range(xv):
		print(f'iteration {x}')

		#splitting in train and test
		#separate for fakes and trues in case we want to have fakes and trues to be a different length

		train_trues, test_trues = [], []
		for i in range(len(trues)):
			if i % xv == x:
				test_trues.append(trues[i])
			else:
				train_trues.append(trues[i])

		train_fakes, test_fakes = [], []
		for i in range(len(fakes)):
			if i % xv == x:
				test_fakes.append(fakes[i])
			else:
				train_fakes.append(fakes[i])
		#print(len(train_trues), len(train_fakes))

		one_dim_table = train_trues + train_fakes
		#print(one_dim_table[0])
		assert(k <= len(one_dim_table))

		#print(len(one_dim_table))

		list_bases = {'A': 1.0, 'C': 2.0, 'G': 3.0, 'T': 4.0}

		converted_sequences = []

		for i in range(len(one_dim_table)):
			converted_sequences.append([])

		for item in range(len(one_dim_table)):
			for i in one_dim_table[item]:
				if i in list_bases.keys():
					converted_sequences[item].append(list_bases[i])

		df = pd.DataFrame(converted_sequences)

		headers = []
		for i in range(len(converted_sequences[0])):
			headers.append(str(f'p{i}'))
		df.columns = headers


		for num in range(2, k+1):
			kmeans = KMeans(num).fit(df)
			centroids = kmeans.cluster_centers_
			#print('Number of clusters:', num, '\n',
				  #'Labels:', kmeans.labels_)
				  #'Centroids:','\n', centroids)

			###TRUES
			true_seq_splitted_by_labels = {}

			labels_true = kmeans.labels_[:len(train_trues)]

			for label, sequence in zip(labels_true, train_trues):
				#print(label, sequence)
				if label in true_seq_splitted_by_labels:
					true_seq_splitted_by_labels[label].append(sequence)
				else:
					true_seq_splitted_by_labels[label] = []
					true_seq_splitted_by_labels[label].append(sequence)
			#print('Complete true:', true_seq_splitted_by_labels)


			###FAKES
			fake_seq_splitted_by_labels = {}

			labels_fake = kmeans.labels_[len(train_trues):]

			for label, sequence in zip(labels_fake, train_fakes):
				if label in fake_seq_splitted_by_labels:
					fake_seq_splitted_by_labels[label].append(sequence)
				else:
					fake_seq_splitted_by_labels[label] = []
					fake_seq_splitted_by_labels[label].append(sequence)
			#print('Complete fake', fake_seq_splitted_by_labels)

			#pwm

			pwm_thr = {}
			pwm_for_true_label = {}

			for true_label, true_list_of_seq in true_seq_splitted_by_labels.items():
				for fake_label, fake_list_of_seq in fake_seq_splitted_by_labels.items():
					if true_label == fake_label: ###change later, because going over the same labels more than once
						#print('checking',true_list_of_seq)
						print('Label_true:', true_label)
						print('Label_fake:', fake_label)
						if true_label in pwm_thr:
							pass
						else:
							pwm_for_true_label[true_label] = make_pwm(true_list_of_seq)
							pwm_thr[true_label] = pwm_threshold(true_list_of_seq, fake_list_of_seq, xv)
							###do i use both to create a threshold?

			#print(pprint.pformat(pwm_thr))

			###delete later if not needed
			#pwm_thr_it[f'iteration {x}'] = pwm_thr
			#pwm_for_true_label_it[f'iteration {x}'] = pwm_for_true_label

			###test
			print('threshold', pwm_thr)
			print('length of test set', len(test_set))
			print('length of fake test', len(test_fakes))
			print('length of true test', len(test_trues))

			test_set = test_trues + test_fakes


			checking_pwm_for_each_label = {}
			for label in range(num):
				checking_pwm_for_each_label[label] = {}
				checking_pwm_for_each_label[label]['trues'] = []
				checking_pwm_for_each_label[label]['fakes'] = []

			for label, pwm_p in pwm_for_true_label.items():
				#tr = 0
				#fl = 0
				for test_seq in test_set:
					score = 1
					for base, probability_of_base in zip(test_seq, pwm_p):
						score = score * probability_of_base[base]

					if score > pwm_thr[label]:
						#tr += 1
						checking_pwm_for_each_label[label]['trues'].append(test_seq)
						###highest score
					elif score <= pwm_thr[label]:
						#fl += 1
						checking_pwm_for_each_label[label]['fakes'].append(test_seq)

				#print('Label', label)
				#print('Trues', tr)
				#print('Fakes', fl)
			#print(checking_pwm_for_each_label)

			print('Number of clusters:', num)
			for label, trues_fakes in checking_pwm_for_each_label.items():
				print('Checking for each label', label)
				t_correct = 0
				t_mistake = 0

				f_correct = 0
				f_mistake = 0
				for true_fake, sequences in trues_fakes.items():
					for seq in sequences:
						if true_fake == 'trues' and seq in test_trues:
							t_correct += 1
						elif true_fake == 'trues' and seq in test_fakes:
							t_mistake += 1
						elif true_fake == 'fakes' and seq in test_fakes:
							f_correct += 1
						elif true_fake == 'fakes' and seq in test_trues:
							f_mistake += 1
				print('Trues placed in trues:', t_correct)
				print('Trues placed in false:', t_mistake)
				print('Accuracy:', t_correct/(t_correct+t_mistake))
				#print(t_correct + t_mistake)
				print('Fakes placed in fakes:', f_correct)
				print('Fakes placed in trues:', f_mistake)
				print('Accuracy:', f_correct/(f_correct+f_mistake))
				#print(f_correct+f_mistake)





	#sys.exit()




		###checking each training sequence by grabbing the seq from test and putting them in the approperiate category
		#aggregating the score

	#print(pwm_for_true_label_it, pwm_thr_it)
	#sys.exit()
	#pwm_thr_it = pprint.pformat(pwm_thr_it)
	#pwm_for_true_label_it = pprint.pformat(pwm_for_true_label_it)





				#print(true_label, fake_label)
				#print(true_list_of_seq, fake_list_of_seq)
						#print(pwm_threshold(true_list_of_seq, fake_list_of_seq, xv))

			#threshhold for where we use fakes and trues of appropriate label
			#split fakes and trues based on the labels, and then pass them to thre threshold

	#print(pwm_thr_it)







			# do the clustering
			# create k PWMs from k clusters
				# find optimal threshold for each PWM
				# score
			# aggregate scores for these PWMs
		# aggregate scores for this k
	# report highest performance

	return

