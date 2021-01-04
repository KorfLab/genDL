genDL
=======

Genomic experiments in Deep Learning. Our goal is to improve upon
typical PWM methods using neural networks and other methods.

## Directory Structure ##

* arch - archive of old stuff, don't modify
* data - raw data files and scripts for managing/generating data files
* docs - tutorial, API
* gendl - shared libraries (include in PYTHONPATH)
* nn - neural network experiments
* pwm - position weight matrix experiments
* test - unit and functional testing
* xtra - uncategorized experiments that may find an eventual home elsewhere

Each of the directories above has its own `REAMDE.md` file that
describes its contnents and intents. An overview is given below.

## The arch Directory ##

This is the previous incarnation of genDL in all its former glory. This
entire directory will be deleted eventually. Please don't make changes
to this directory ever.

## The data Directory ##

If data is used by more than one experiment, place it in the `data`
directory. Scripts used to generate data may also be placed in `data`.

## The gendl Directory ##

All shared libraries go in the `gendl` directory. Don't put shared
libraries inside experiment directories. All imports should come from
the parent `gendl` namespace followed by the library name. Here are some
examples of how to import gendl libraries.

	import gendl.pwm
	import gendl.pwm as pwm
	from gendl.pwm import read_transfac

## Experiment Directories ##

Experiment directories contain a mixture of code, data, and maybe
figures or other files. Each type of experiment may have several
flavors. For example, you will find `pwm/base` for standard PWM
experiments as well as `pwm/kmeans` for the k-means implementation. Each
sub-experiment should have its own `README.md` to describe its contents
and intents.

	pwm
		base
		kmeans

	nn
		mlp
		cnn

	exp
		edit_distance
		whatever

