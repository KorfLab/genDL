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

	python3 rlooper.py --fasta some-fasta-file

The data output of the program streams json to stdout. The output
directory contains the original rlooper outputs, which you don't need
unless you want to do some error checking. 

## The prep directory ##

There are two kinds of data for the R-loop project

	+ Footprint - peaks inferred from PacBio bisulfite sequencing
	+ DRIPc, sDRIP, and qDRIP - peaks inferred from Illumina reads

Data preparation requires the human genome file in 2bit format as well
as a program to decode/retrieve sequences, and a bigwig decoder.
`hg19.2bit`, `twoBitToFa`, and `bigWigToWig` can be downloaded from the
UCSC genome site. Soft-link all of these to the prep directory.

Note that if you are on a Mac and you download a binary, you won't have
permission to run it unless you remove the security protection.

	xattr -c twoBitToFa bigWigToWig
	
### SMRT-seq footprint data ###

PEAKS_GENOME is a directory of R-loops inferred from bisulfite PacBio
reads. There are 30 genes with multiple reads per gene.

To convert the original data to fasta:

	pacbio2fasta.py > peaks.fa

The format of the fasta headers is the following

	+ gene
	+ read identifier (from original data)
	+ chromosome
	+ begin
	+ end
	+ strand

### DRIPc, sDRIP, and qDRIP data ##

The `info.txt` file contains a high quality set of data curated by the Chedin lab. The original data was provided at the following link.

https://docs.google.com/spreadsheets/d/1xyelh5_rKflwZJ2lD-biykcn9kcuWfGOV2bOy2zJy58/edit?usp=sharing

To retrieve all files use the download script.

	perl downloader.py




