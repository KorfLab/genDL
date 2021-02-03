import sys
import pandas as pd
import numpy as np
from gendl import pwm, seqio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from apyori import apriori

def conv_data(seqs):
	####RELOCATE TO gendl
	converting = {'A':1.0, 'C':2.0, 'G':3.0, 'T':4.0}

	df = []
	for seq in seqs:
		conv_seq = []
		for base in seq:
			if base in converting:
				conv_seq.append(converting[base])
		df.append(conv_seq)

	df = pd.DataFrame(df)
	return df

def sorting(seqs, kmeans, k):
	sort_by_label = {}
	for label, seq in zip(kmeans.labels_, seqs):
		if label not in sort_by_label:
			sort_by_label[label] = []
		sort_by_label[label].append(seq)
	assert(len(sort_by_label.keys()) == k)

	return sort_by_label

def acc(seqs, mpwm, npwm):
	match = 0
	nmatch = 0
	for seq in seqs:
		m_score = 0
		for m_label, m_pwm in mpwm.items():
			mout = pwm.score_pwm(m_pwm, seq)
			if mout > m_score:
				m_score = mout

		n_score = 0.0
		for n_label, n_pwm in npwm.items():
			nout = pwm.score_pwm(n_pwm, seq)
			if nout > n_score:
				n_score = nout

		if m_score > n_score:
			match += 1
		elif n_score > m_score:
			nmatch +=1
	return match, nmatch

def pca_kmeans(df, seqs, k):
	#extracting values from dataframe and standardizing
	val = df.values
	#transform data such that its distribution has a mean value 0 and standard deviation of 1
	val_std = StandardScaler().fit_transform(df)

	#eigenvalues and eigen vectors
	mean_vec = np.mean(val_std, axis = 0)
	cov_mat = np.cov(val_std.T)
	eig_vals, eig_vecs = np.linalg.eig(cov_mat)

	#Making a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

	#sorting pairs in descending order
	eig_pairs.sort(key= lambda x: x[0], reverse = True)
	tot = sum(eig_vals)
	var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse = True)]
	#cumulative sum of the elements along a given axis
	cum_var_exp = np.cumsum(var_exp)

	#plotting (eigenvalue, eigenvector) tuples
	plt.figure(figsize = (10,5))
	plt.bar(range(len(seqs[0])), var_exp, alpha = 0.3333, align = 'center', color = 'gold')
	plt.ylabel('explained_variance_ratio')
	plt.xlabel('Principal Components')
	plt.title('Percentage of Variance (Information) for each by PC')
	plt.show()

	#PCA Analysis
	pca = PCA(n_components = len(seqs[0]))
	x_9d = pca.fit_transform(val_std)

	#visualization with k-means clustering
	kmeans = KMeans(k)
	label_color_map = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c', 5: 'm'}
	x_clustered = kmeans.fit_predict(x_9d)
	label_color = [label_color_map[i] for i in x_clustered]

	plt.scatter(x_9d[:,0], x_9d[:,1], c = label_color, alpha = 0.5)
	plt.ylabel('Principal Component 2')
	plt.xlabel('Principal Component 1')
	plt.title(' PC2 vs PC1: PCA applied to Kmeans Clustering')
	plt.show()

#used in apriori
def list_position(seqs, start, stop):
	####RELOCATE TO gendl
	assert(start <= stop)
	total = []
	for seq in seqs:
		single_seq = []
		for base_pos in range(len(seq)):
			single_seq.append(f'{seq[base_pos]}{base_pos+start}')
		assert(len(single_seq) == stop-start)
		total.append(single_seq)
	return total

#specific to apriori
def appr(seqs, start, stop, min_sup):
	conv_seqs = list_position(seqs, start, stop)
	seq_rules = apriori(conv_seqs, min_support = min_sup)
	seq_results = list(seq_rules)

	support_values = [item[1] for item in seq_results]
	listRules = [list(seq_results[i][0]) for i in range(0, len(seq_results))]
	assoc_rules = []
	for rules, value in zip(listRules, support_values):
		if value < 1.0:
			assoc_rules.append([tuple(rules), value])
	return assoc_rules

#ask if there is a better way of doing that
def preping_for_pwm(data, rule):
	match = []
	nmatch = []
	for seq in data:
		if set(rule[0]).issubset(seq) == True:
			match.append(seq)
		elif set(rule[0]).issubset(seq) == False:
			nmatch.append(seq)
	assert(len(match)+len(nmatch) == len(data))

	updated_match = []
	for seq in match:
		old = ''
		for base in seq:
			for char in base:
				if char.isalpha():
					old += char
		updated_match.append(old)

	updated_nmatch = []
	for seq in nmatch:
		old = ''
		for base in seq:
			for char in base:
				if char.isalpha():
					old += char
		updated_nmatch.append(old)

	match = updated_match
	nmatch = updated_nmatch

	#https://spapas.github.io/2016/04/27/python-nested-list-comprehensions/
	#match = [''.join(filter(str.isalpha, i)) for i in match]
	return(match, nmatch)

def filtering(rules1, rules0):
	dict_rules1 = {}
	dict_rules0 = {}

	#converting to dictionary
	for rule1, value1 in rules1:
		dict_rules1[rule1] = value1
	for rule0, value0 in rules0:
		dict_rules0[rule0] = value0
	#filtering
	for rule1, value in rules1:
		if rule1 in dict_rules0:
			del dict_rules0[rule1]
			del dict_rules1[rule1]

	#converting to the original format
	updated_rules1 = []
	updated_rules0 = []

	for key1, value1 in dict_rules1.items():
		updated_rules1.append([key1, value1])

	for key, value in dict_rules0.items():
		updated_rules0.append([key, value])

	return updated_rules1, updated_rules0

def acc_appr(train_set, rule, test):
	match, nmatch = preping_for_pwm(train_set, rule)
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

#specific to kmeans script
def regroup(labels, seqs):
	#predicting labels of test data
	summary = {}
	for label, test in zip(labels, seqs):
		if label not in summary:
			summary[label] = []
			summary[label].append(test)
		else:
			summary[label].append(test)

	#sorting by labels and appending to the total
	perc = []
	for label, seqs in summary.items():
		count = 0
		for seq in seqs:
			if seq[0] == 1:
				count += 1
		perc.append(count/len(seqs))
	perc = sorted(perc, reverse = True)

	return perc

#specific to kmeans script
def outcome(distr, k):
	total = []
	for i in range(k):
		lst = [item[i] for item in distr]
		lst = f'{sum(lst)/len(lst):.4f}'
		total.append(lst)

	return (print(', '.join(total)))



