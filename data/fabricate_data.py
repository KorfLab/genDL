
import argparse
import random
import sys


parser = argparse.ArgumentParser(
	description='Generate fabricated data for splice donors and acceptors')
parser.add_argument('--seqs', required=False, type=int, default=10000,
	metavar='<int>', help='number of sequences [%(default)i]')
parser.add_argument('--seed', required=False, type=int,
	metavar='<int>', help='use a specific random seed')
arg = parser.parse_args()

if arg.seed: random.seed(arg.seed)
seqs = arg.seqs

## Ian's arcane syntax for creating fabricated data

don_not = 'NNNNNNNNNNNNNNNNNNNNGTNNNNNNNNNNNNNNNNNNNN'
don_obs = 'NNNNNNNNNNNNNNNNNNNNGTrrgNNNNNNNNNNNNNNNNN'
don_ex1 = 'NNNNNNNNNNNNNNNNNNNNGT1.gNNNNNNNNNNNNNNNNN'
don_ex2 = 'NNNNNNNNNNNNNNNNNNNNGT2.gNNNNNNNNNNNNNNNNN'

acc_not = 'NNNNNNNNNNNNNNNNNNNNAGNNNNNNNNNNNNNNNNNNNN'
acc_obs = 'NNNNNNNNNNNNNNNNtttcAGNNNNNNNNNNNNNNNNNNNN'
acc_ex3 = 'NNNNNNNNNNNNNNNN3...AGNNNNNNNNNNNNNNNNNNNN'
acc_ex4 = 'NNNNNNNNNNNNNNNN4...AGNNNNNNNNNNNNNNNNNNNN'
acc_ex5 = 'NNNNNNNNNNNNNNNN5...AGNNNNNNNNNNNNNNNNNNNN'
acc_ex6 = 'NNNNNNNNNNNNNNNN6...AGNNNNNNNNNNNNNNNNNNNN'

sigs = {
	'N' : (1, 'ACGT'), # 25% each nucleotide
	'A' : (1, 'A'),
	'C' : (1, 'C'),
	'G' : (1, 'G'),
	'T' : (1, 'T'),
	'a' : (1, 'AAAAAAAAAAAAAAAAACGT'), # 85% A
	'c' : (1, 'CCCCCCCCCCCCCCCCCAGT'), # 85% C
	'g' : (1, 'GGGGGGGGGGGGGGGGGACT'), # 85% G
	't' : (1, 'TTTTTTTTTTTTTTTTTACG'), # 85% T
	'r' : (1, 'AAAAAAAAAGGGGGGGGGCT'), # 90% A or G
	'y' : (1, 'CCCCCCCCCTTTTTTTTTAG'), # 90% C or T
	'1' : (2, ['AA', 'GG']),           # perfect proximal linkage
	'2' : (2, ['aa', 'gg']),           # imperfect proximal linkage
	'3' : (4, ['TTTC', 'CTTT']),       # perfect distal linkage
	'4' : (5, ['tttc', 'cttt']),       # imperfect distal linkage
	'5' : (5, ['TTNC', 'NTTC']),       # perfect 2 modes
	'6' : (5, ['ttNc', 'Nttc']),       # imperfect 2 modes
}

def create_file(sig, filename, count):
	with open(f'{filename}.fa', 'w') as fp:
		for c in range(count):
			fp.write(f'>{filename}.{c}\n')
			for s in sig:
				if s == '.': continue
				n, choice = sigs[s]
				if n == 1: fp.write(random.choice(choice))
				else:
					string = random.choice(choice)
					for nt in string:
						n, choice = sigs[nt]
						fp.write(random.choice(choice))
			fp.write('\n')


## Outputs

create_file(don_not, 'don.not', arg.seqs)
create_file(don_obs, 'don.obs', arg.seqs)
create_file(don_ex1, 'don.ex1', arg.seqs)
create_file(don_ex2, 'don.ex2', arg.seqs)

create_file(acc_not, 'acc.not', arg.seqs)
create_file(acc_obs, 'acc.obs', arg.seqs)
create_file(acc_ex3, 'acc.ex3', arg.seqs)
create_file(acc_ex4, 'acc.ex4', arg.seqs)
create_file(acc_ex5, 'acc.ex5', arg.seqs)
create_file(acc_ex6, 'acc.ex6', arg.seqs)