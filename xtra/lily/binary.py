#!/usr/bin/python3

import gzip

def convert_to_binary(file):
	# seq as input
	#read fasta
	with gzip.open(file,'rt') as fp:
		for line in fp.readlines():
			line = line.strip()
			if line.startswith('>'):
				continue
			else:
				
			print(line)
		#line = fp.readlines()
		
		#print(line)
		
file = '../../data/acc.ex4.fa.gz'
(convert_to_binary(file))