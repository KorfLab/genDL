import gzip
import os
import sys

from gendl.seqio import read_fasta
from grimoire.toolbox import revcomp_str

# read the original data from Fred/Stella
base = 'PEAKS_GENOME'
peaks = []
for file in os.listdir(base):
	col = file.split('_')
	gene = col[1][4:]
	with gzip.open(f'{base}/{file}', 'rt') as fp:
		for line in fp.readlines():
			(chr, beg, end, name, z, strand, foo) = line.split()
			col = name.split('/')
			rid = col[1]
			peaks.append((gene, rid, chr, int(beg), int(end), strand))

# get sequences using twoBitToFa (which doesn't do reverse-complement)
with open('peaks.tmp', 'w') as fp:	
	for peak in peaks:
		(gene, rid, chr, beg, end, strand) = peak
		beg -= 1
		fp.write(f'{chr}:{beg}-{end}\n')
os.system('twoBitToFa hg19.2bit fa.tmp seqList=peaks.tmp')

# fix fasta file
for i, (name, seq) in enumerate(read_fasta('fa.tmp')):
	(gene, rid, chr, beg, end, strand) = peaks[i]
	seq = seq.upper()
	if strand == '-': seq = revcomp_str(seq)
	print(f'>{gene}-{chr}:{beg}-{end}:{strand}')
	for i in range(0, len(seq), 60):
		print(seq[i:i+60])

# clean up
os.system('rm peaks.tmp fa.tmp')
