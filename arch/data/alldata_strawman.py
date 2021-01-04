#!/usr/bin/env python3

import argparse
import sys
import re
import math
import gzip
import strawman

parser = argparse.ArgumentParser(
	description='PWM classifier using all splice data')
parser.add_argument('--true', required=True, type=str,
	metavar='<path>', help='.gz file of true sequences')
parser.add_argument('--fake', required=True, type=str,
	metavar='<path>', help='.gz file of fake sequences')

arg = parser.parse_args()

def read_data(filename):
	fp = None
	seqs = []
	if filename.endswith('.gz'):
		fp = gzip.open(filename, 'rt')
	else:
		fp = open(filename)

	for line in fp.readlines():
		seqs.append(line.rstrip())
		
	fp.close()
	
	return seqs

trues = read_data(arg.true)
fakes = read_data(arg.fake)
print(len(fakes))
print(len(trues))
counter = 0
ACC = 0
for i in range(0,len(fakes)-len(trues)+1,len(trues)):
	counter += 1
	print(counter, i, i+len(trues)-1)
	acc = strawman.pwm_vs_pwm(trues, fakes[i:i+len(trues)], 10)
	print(f"{acc:.4f}")
	ACC += acc

print(f"{ACC/counter:.4f}")