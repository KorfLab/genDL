README for data directory
=========================

## Fabricated Data ##

The script `fabricate_data.py` creates fabricated splice data.

+ _don_ means donor site
+ _acc_ means acceptor site
+ _obs_ means looks like observed site
+ _not_ means looks like not observed site
+ _ex#_ means experiment number (see below)

The following command was used to generate data (2021-01-04)

	python3 fabricate_data.py --seed 1
	gzip acc.*.fa don.*.fa

## Experiment Numbers ##

The base donor concensus is GTrrg. That means that the GT will be
produced 100% of the time and the other letters will follow purine,
purine, guanine with some mismatches allowed (see code).

+ ex1 - donor sites where the rrg is AAg or GGg (perfect A and G)
+ ex2 - donor sites where the rrg is aag or ggg (as above but imperfect)

The base acceptor concensus is ytttyAG. The AG is produced 100% of the
time, but the other positions are allow mismatches.

+ ex3 - acceptor sites with CTTTC or TTTTT (perfect)
+ ex4 - acceptor sites with ctttc or ttttt (imperfect)
+ ex5 - acceptor sites with TTTNC or NTTTC (perfect, but slides 1 bp)
+ ex6 - acceptor sites with tttNc or Ntttc (as above, but imperfect)


