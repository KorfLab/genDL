
import random

def randseq(length, a, c, g, t):
	seq = ''
	for i in range(length):
		r = random.random()
		if r < a: seq += 'A'
		elif r < a + c: seq += 'C'
		elif r < a + c + g: seq += 'G'
		else: seq += 'T'
	return seq

def edit_dist(s1, s2):
	d = 0
	for i in range(len(s1)):
		if s1[i] != s2[i]: d += 1
	return d


hist = [0] * 41
sum = 0
size = 100000000
for i in range(size):
	s1 = randseq(20, 0.3, 0.2, 0.2, 0.3) + randseq(20, 0.2, 0.3, 0.3, 0.2)
	s2 = randseq(20, 0.3, 0.2, 0.2, 0.3) + randseq(20, 0.2, 0.3, 0.3, 0.2)
	
	d = edit_dist(s1, s2)
	hist[d] += 1
	sum += d

print(f'ave {sum/size}')
for i in range(len(hist)):
	print(i, hist[i])