
import argparse
import gzip
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--file1', required=True, type=str,
	metavar='<path>', help='path to _true_ sequence file')
parser.add_argument('--file2', required=True, type=str,
	metavar='<path>', help='path to _fake_ sequence file')
parser.add_argument('--count1', required=True, type=int,
	metavar='<int>', help='length of sequence')
parser.add_argument('--count2', required=True, type=int,
	metavar='<int>', help='length of sequence')
parser.add_argument('--offset', required=True, type=int,
	metavar='<int>', help='left-hand offset')
parser.add_argument('--length', required=True, type=int,
	metavar='<int>', help='length of sequence')
parser.add_argument('--fixseed', required=False, action='store_true',
	help='length of sequence')
arg = parser.parse_args()

def readseq(path, o, l):
	seqs = []
	with gzip.open(path, 'rt') as fp:
		for line in fp.readlines():
			seq = []
			for i in range(o, o+l):
				seq.append(line[i])
				nt = line[i]
			seqs.append(seq)
	random.shuffle(seqs)
	return seqs

def seq2hot(seqs, label, n):
	output = []
	for s in seqs:
		val = []
		for nt in s:
			if   nt == 'A': val.append('1,0,0,0')
			elif nt == 'C': val.append('0,1,0,0')
			elif nt == 'G': val.append('0,0,1,0')
			else:           val.append('0,0,0,1')
		val.append(label)
		output.append(','.join(val))
		if len(output) == n:
			return output

if __name__ == '__main__':

	if arg.fixseed: random.seed(1)

	true = readseq(arg.file1, arg.offset, arg.length)
	fake = readseq(arg.file2, arg.offset, arg.length)

	assert(len(true) >= arg.count1)
	assert(len(fake) >= arg.count2)

	t = seq2hot(true, 't', arg.count1)
	f = seq2hot(fake, 'f', arg.count2)

	all = t + f
	random.shuffle(all)
	for line in all: print(line)
	