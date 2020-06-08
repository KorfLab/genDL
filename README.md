genDL
=======

Genomic experiments in Deep Learning

## Data ##

The `data` directory contains the following files

* `data_organizer.py` script that created the data files below
* `don.fake.txt.gz` fake donor sites
* `don.hi.true.txt.gz` true donor sites with common usage
* `don.lo.true.txt.gz` true donor sites with rare usage
* `acc.fake.txt.gz` fake acceptor sites
* `acc.hi.true.txt.gz` true acceptor sites with common usage
* `acc.lo.true.txt.gz` true acceptor sites with rare usage

Data was processed 2020-06-08 with WormBase version 276. The length of
the donor and acceptor sites flanks was set at 20. That means each
sequence is 42 bp long. The threshold for 'common' vs 'rare' is an
RNASeq_splice score of 1000. These are the default values in the
data_organizer.py script. A _fake_ donor or acceptor site is any
transcribed sequence that contains the canonical GT or AG signal but has
not been seen in the RNASeq_splice data.

All of the data comes from the + strand. This leaves the - strand as the
testing set.

| File            | Sequences |
|:----------------|:----------|
| acc.fake.txt    |  1663602  |
| acc.lo.true.txt |   231802  |
| acc.hi.true.txt |    54672  |
| don.fake.txt    |  1543819  |
| don.lo.true.txt |   243584  |
| don.hi.true.txt |    54787  |

