import argparse
import random
from gendl import seqio


parser = argparse.ArgumentParser(
	description='Generate some random sequences')
parser.add_argument('--file', required=True, type=str,
	metavar='<file>', help='fasta file source')
parser.add_argument('--count', required=True, type=int,
	metavar='<int>', help='number of sequences to generate')
parser.add_argument('--length', required=True, type=int,
	metavar='<int>', help='length of sequences')
parser.add_argument('--seed', required=False, type=int,
	metavar='<int>', help='random seed')
arg = parser.parse_args()

if arg.seed: random.seed(arg.seed)

nt = '' # omg this is so wasteful :)
for name, seq in seqio.read_fasta(arg.file):
	nt += seq

for i in range(arg.count):
	seq = ''
	for j in range(arg.length):
		seq += random.choice(nt)
	print(f'>seq-{i}\n{seq}')
