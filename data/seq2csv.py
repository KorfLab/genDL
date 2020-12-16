
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
parser.add_argument('--onehot', required=False, action='store_true',
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

def seq2int(seqs, label, n):
	count = 0
	for s in seqs:
		val = [label]
		for nt in s:
			if   nt == 'A': val.append('1')
			elif nt == 'C': val.append('2')
			elif nt == 'G': val.append('3')
			else:           val.append('4')
		count += 1
		print(','.join(val))
		if count == n: return
		
def seq2hot(seqs, label, n):
	count = 0
	for s in seqs:
		val = [label]
		for nt in s:
			if   nt == 'A': val.append('1,0,0,0')
			elif nt == 'C': val.append('0,1,0,0')
			elif nt == 'G': val.append('0,0,1,0')
			else:           val.append('0,0,0,1')
		count += 1
		print(','.join(val))
		if count == n: return

true = readseq(arg.file1, arg.offset, arg.length)
fake = readseq(arg.file2, arg.offset, arg.length)

assert(len(true) >= arg.count1)
assert(len(fake) >= arg.count2)

if arg.onehot:
	seq2hot(true, '1', arg.count1)
	seq2hot(fake, '0', arg.count2)
else:
	seq2int(true, '1', arg.count1)
	seq2int(fake, '0', arg.count2)

	