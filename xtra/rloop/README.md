README for R-loop experiments
=============================

This is currently in the experimental stages.

## Install rlooper ##

You must first install rlooper.

	git clone https://github.com/chedinlab/rlooper.git
	cd rlooper
	make all

This program has an unusual CLI with no usage statement. Because of
this, don't run this program directly. Instead, use the python wrapper
below. Note that you still need to build the software and make sure the
program is in your executable path.

	PATH=$PATH:wherever-you-installed/rlooper/bin/rlooper

You should probably put that in your .profile, .bash_rc, or whatever.

## Use python wrapper ##

Here's the default command line. There are a bunch of options. There are
also options in the program that aren't available in the wrapper (yet).

	python3 rlooper.py --fasta some-fasta-file --out some-directory

The data output of the program streams a json file to stdout. The output
directory contains the original rlooper outputs, which you don't need
unless you want to do some error checking. 
