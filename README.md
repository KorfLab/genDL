genDL
=======

Genomic experiments in Deep Learning

## Directory Structure ##

* arch - archive of old stuff which may get removed
* data - raw data files and scripts for managing/generating data files
* docs - tutorial, API
* gendl - shared libraries (include in PYTHONPATH)
	* pwmlib
	* kpwmlib
* pwm - programs and experiments for position weight matrix methods
	* base - standard PWM, strawman
	* kpwm - k-means PWM (PCA and supporting experiments)
* nn - multilayer perceptron methods
	* mlp
	* cnn
* xtra 
	* tree - tree-based method
	* mdd
* test - eventually for unit and functional testing

----

Stuff below needs reorganization for the structure above

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

| File                | Sequences |
|:--------------------|:----------|
| acc.fake.txt.gz     | 186671    |
| acc.hi.true.txt.gz  | 8193      |
| acc.lo.true.txt.gz  | 9148      |
| don.fake.txt.gz     | 172559    |
| don.hi.true.txt.gz  | 8215      |
| don.lo.true.txt.gz  | 8708      |

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

### Metrics ###
We are using average of true positive rate and positive predictive value. 

## PWM Strawman ## 

We learned a PWM from sequence data to classify the different sites and their 
usage. `strawman.py` performs those experiments. 

Results
| acc/don | task         | n     | xv | PWM    | PWM vs PWM |
|:------- |:------------ |:----- |:-- |:------ |:---------- |
| acc     | hi-vs-fake   | 8000  | 10 | 0.9248 | 0.9408     |
| acc     | lo-vs-fake   | 9000  | 10 | 0.7576 | 0.7293     |
| acc     | hi-vs-lo     | 8000  | 10 | 0.8335 | 0.8348     |
| acc     | hilo-vs-fake | 17000 | 10 | 0.7903 | 0.8000     |
| don     | hi-vs-fake   | 8000  | 10 | 0.8717 | 0.9199     |
| don     | lo-vs-fake   | 8000  | 10 | 0.7488 | 0.7421     |
| don     | hi-vs-lo     | 8000  | 10 | 0.7892 | 0.7793     |
| don     | hilo-vs-fake | 17000 | 10 | 0.7694 | 0.8150     |


## Data preparation ##

To prepare the data for training deep neural networks, sequences first need
to be one-hot encoded and shaped into proper numpy arrays. In `data` directory 
is the `data_shaper.py` script that does the pre-processing. It is required to
build a `one_hot/` directory to save the one-hot encoded data.

## DL Model Training ##

Currently using `eval.py` in `train/`. Construct your model there, and it will
produce the accuracy from validation set. 
*TO-DO*
+ Allow for cross validation.

## DL Architectures ##

1. Feed forward neural network with flatten one-hot encoded inputs. 
	+ Hyperparameters (network)
		+ Number of layers
		+ Number of units per layer
		+ Regularization per layer
		+ Dropout per layer
		+ Activation functions
	+ Hyperparameters (optimization)
		+ Gradient optimizers
		+ Learning rate
		+ Learning rate schedule
		+ Batch size
		+ Epochs
2. Convolutional Neural network with one-hot encoded matrix inputs
	+ Hyperparameters (network)
		+ Number of convolutional layers
		+ Number of pooling layers
		+ Filter sizes
		+ Filter strides
	+ Same optimization hyperparameters

## Model Optimization ##

Still in development. 
Trying to make a class structure to produce model objects that can be iterated 
over to test many hyperparameters. We also want to try and use the hparams
module in tensorboard. 

## Model Deployment ##

Trained models are deployed across _C. elegans_ genome to find splice site
donor and acceptor sites. Either deploy models on entirety of genome, and/or
to specific gene regions of the genome. Applying models to regions with no
genes may result in predictions of splice sites where there are none (be
interesting to test). 

The way deploymed models should be leveraged is using reciprocal predictions.
A hi splice is called when it is predicted as hi label from hi vs fake model as
well as hi vs lo model. Reciprocal prediction may not be the proper term, but
the idea is that a label to be called (label meaning acc.hi don.hi etc) needs to
be predicted from the two appropriate models. 

Ideally, the advantage of this would be we could increase specificity. A hi acc
site needs to be predicted from both hi-vs-fake and hi-vs-lo models. However,
this approach could have disadvantages. One being loss of sensitivity. Another
being a multiple comparisons problem. I am not sure what the statistical issues
we introduce when doing two predictions. 




























