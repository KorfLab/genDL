import sys
import gzip

NT = ['A', 'C', 'G', 'T']
beg = 16
end = 24
for i in range(beg, end, 1):
	for nt in NT:
		set1 = []
		set2 = []
		with gzip.open(sys.argv[1], 'rt') as fp:
			for line in fp.readlines():
				if line[i] == nt: set1.append(line)
				else:             set2.append(line)
		if len(set1) == 0 or len(set2) == 0: continue
		with open(f'set-{i}{nt}+.txt', 'w') as fp:
			for line in set1:
				fp.write(line)
		with open(f'set-{i}{nt}-.txt', 'w') as fp:
			for line in set2:
				fp.write(line)
				
