import argparse
import csv
import random

def read_file(filename):
	data = []
	with open(filename) as fp:
		fp.readline()
		reader = csv.reader(fp)
		for row in reader:
			survive = True if row[1] == '1' else False # survival
			female = True if row[4] == 'female' else False # gender
			pclass = int(row[2]) # passenger class
			age = 0.0 if row[5] == '' else float(row[5]) # age
			name = row[3] # name
			data.append({'survive': survive, 'female': female, 'age': age,
				'name': name, 'pclass': pclass})
	return data


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='strawman titanic')
	parser.add_argument('--file', required=True, type=str,
		metavar='<path>', help='path to cvs file (train.csv)')
	parser.add_argument('--fold', required=False, type=int, default=5,
		metavar='<int>', help='cross validation-ish')
	parser.add_argument('--split', required=False, type=float, default=0.5,
		metavar='<float>', help='fraction of set')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='random seed')
	arg = parser.parse_args()

	if arg.seed: random.seed(arg.seed)

	all = read_file(arg.file)
	
	for i in range(arg.fold):
		test = []
		for p in all:
			if random.random() < arg.split: test.append(p)
		
		# female model (females survive)
		tp = 0
		tn = 0
		fp = 0
		fn = 0
		for p in test:
			if       p['female'] and     p['survive']: tp += 1
			elif     p['female'] and not p['survive']: fp += 1
			elif not p['female'] and     p['survive']: fn += 1
			elif not p['female'] and not p['survive']: tn += 1
		
		acc = (tp + tn) / (tp + tn + fp + fn)
		print('female', tp, tn, fp, fn, acc)
		
		# pclass model (first class survives)
		tp = 0
		tn = 0
		fp = 0
		fn = 0
		for p in test:
			if   p['pclass'] == 1 and     p['survive']: tp += 1
			elif p['pclass'] == 1 and not p['survive']: fp += 1
			elif p['pclass'] != 1 and     p['survive']: fn += 1
			elif p['pclass'] != 1 and not p['survive']: tn += 1
		
		acc = (tp + tn) / (tp + tn + fp + fn)
		print('pclass', tp, tn, fp, fn, acc)
		
		# name model (longer name survives)
		s = 25
		tp = 0
		tn = 0
		fp = 0
		fn = 0
		for p in test:
			if   len(p['name']) > s  and     p['survive']: tp += 1
			elif len(p['name']) > s  and not p['survive']: fp += 1
			elif len(p['name']) <= s and     p['survive']: fn += 1
			elif len(p['name']) <= s and not p['survive']: tn += 1
		
		acc = (tp + tn) / (tp + tn + fp + fn)
		print('name', tp, tn, fp, fn, acc)
		
		# age model (younger survive)
		n = 20
		tp = 0
		tn = 0
		fp = 0
		fn = 0
		for p in test:
			if   p['age'] < n  and     p['survive']: tp += 1
			elif p['age'] < n  and not p['survive']: fp += 1
			elif p['age'] >= n and     p['survive']: fn += 1
			elif p['age'] >= n and not p['survive']: tn += 1
		
		acc = (tp + tn) / (tp + tn + fp + fn)
		print('age', tp, tn, fp, fn, acc)
		
		print()
		