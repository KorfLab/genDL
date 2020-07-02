genDL
=======

Genomic experiments in Deep Learning

## Data ##

The `data` directory contains the following data files

* `don.fake.txt.gz` fake donor sites
* `don.hi.true.txt.gz` true donor sites with common usage
* `don.lo.true.txt.gz` true donor sites with rare usage
* `acc.fake.txt.gz` fake acceptor sites
* `acc.hi.true.txt.gz` true acceptor sites with common usage
* `acc.lo.true.txt.gz` true acceptor sites with rare usage

Data was processed 2020-06-30 with WormBase version 276. The length of
the donor and acceptor sites flanks was set at 20. That means each
sequence is 42 bp long. Collected sequences represent high-confidence
donor/acceptor sites. **(More details here)**. Script to produce the dataset
is in the Lyman2020 repo **(correct?)**. A _fake_ donor or acceptor
site is any transcribed sequence that contains the canonical GT or AG signal
but has not been seen in the RNASeq_splice data. Data comes from both
positive and negative strands.

| File            | Sequences |
|:----------------|:----------|
| acc.fake.txt    |  210674   |
| acc.lo.true.txt |   10432   |
| acc.hi.true.txt |    9196   |
| don.fake.txt    |  194418   |
| don.lo.true.txt |    9862   |
| don.hi.true.txt |    9220   |

## Data preparation ##

To prepare the data for training deep neural networks, sequences first need
to be one-hot encoded and shaped into proper numpy arrays. In `data` directory 
is the `data_shaper.py` script that does the pre-processing. It is required to
build a `one_hot/` directory to save the one-hot encoded data.

*TO-DO*
+ edit data_shaper.py to include unique identifiers for pickeled datasets.  

## DL Model Training ##

Currently using `eval.py` in `train/`. Construct your model there, and it will
produce the accuracy from validation set. 
*TO-DO*
+ Allow for cross validation. 

## Dataset subsetting ##

`acceptors`
| Level | Training | Validation |
| :---- | :------- | :--------- |
| hi    | 9000     | 196        |
| lo    | 10000    | 432        |
| fake  | 210000   | 674        |

`donors`
| Level | Training | Validation |
| :---- | :------- | :--------- |
| hi    | 9000     | 220        |
| lo    | 9000     | 862        |
| fake  | 194000   | 418        |

Example, for hi vs fake training, train on all non-overlapping combinations
of 9000 hi with 9000 fake. Each 9000x9000 training set is cross-validated. 
Extra held out validation is used for post-validation. 
9000 vs 210000 is 23 models to be cross validated. 

## Learning tasks ##

For both acceptors and donors, we want to learn how to classify 4 types of
binary labels. 

1. hi vs fake # id high usage sites
2. lo vs fake # id low usage sites
3. hi vs lo   # distinguish between hi and lo
4. hi+lo vs fake # distinguish fake from combination of hi+lo sequences

Validation sets for each task. 
1. validation hi and fakes plus the lo 
2. validation lo and fakes plus the hi
3. validation hi and lo plus fakes
4. validation hi lo and fakes

Total 8 learning tasks, the 4 listed for both acceptors and donors.

## Position weight matrix strawman ##

perform xv on a specific true/fake set




















































