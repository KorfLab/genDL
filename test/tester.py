
import gendl.seqio
import gendl.pwm

# testing fasta
for d, s in gendl.seqio.read_fasta('foo.fasta'):
	print(d, s)

# testing raw
seqs = []
for s in gendl.seqio.read_raw('raw.txt.gz'):
	print(s)
	seqs.append(s)

# create pwm
pwm = gendl.pwm.make_pwm(seqs)
print(pwm)

# score pwm
for seq in seqs:
	print(seq, gendl.pwm.score_pwm(seq, pwm))

# some fake data
fake = []
for i in range(5):
	fake.append(gendl.seqio.random_dna(len(pwm)))
print(fake)
fwm = gendl.pwm.make_pwm(fake)

# test threshold scoring
threshold = 0.1
print(gendl.pwm.pwm_vs_threshold(pwm, threshold, seqs, fake))

# test pwm_vs_pwm
print(gendl.pwm.pwm_vs_pwm(pwm, fwm, seqs, fake))


