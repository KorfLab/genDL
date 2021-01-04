
import random
import sys

seqs = 5000


N = 'ACGT'
G = 'G'
T = 'T'
r = 'AG'
g = 'ACGGGGGGGT'
m = ['AA', 'GG']

# fake donor sites
fake = []
for i in range(20): fake.append(N)
fake.append(G)
fake.append(T)
for i in range(20): fake.append(N)
with open('fab.don.fake.txt', 'w') as fp:
	for i in range(seqs):
		s = ''
		for j in range(len(fake)):
			s += random.choice(fake[j])
		fp.write(f'{s}\n')

# true donor sites no dependency
true = []
for i in range(20): true.append(N)
true.append(G)
true.append(T)
true.append(r)
true.append(r)
true.append(g)
for i in range(17): true.append(N)

with open('fab.don.true.txt', 'w') as fp:
	for i in range(seqs):
		s = ''
		for j in range(len(true)):
			s += random.choice(true[j])
		fp.write(f'{s}\n')

# true donors sites with 100% dependancy
link = []
for i in range(20): link.append(N)
link.append(G)
link.append(T)
link.append(m)
link.append(g)
for i in range(17): link.append(N)

with open('fab.don.link.txt', 'w') as fp:
	for i in range(seqs):
		s = ''
		for j in range(len(link)):
			s += random.choice(link[j])
		fp.write(f'{s}\n')
