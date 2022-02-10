import argparse
import numpy as np
import random

## CLI
parser = argparse.ArgumentParser(description="shift small motif")
parser.add_argument("input", type=str, metavar='<file>',
                    help="Enter name of fasta file")
args = parser.parse_args()

out_filename = args.input[:8] + "rearranged.txt"
f = open(out_filename, 'w')

seqs = open(args.input, 'r')
for line in seqs:
    line = line.strip()
    K = round(random.random()*10)
    #print(K) # Check that integers are between 1 and 10
    new_str = line[-K-1:] + line[:-K-1] + '\n'
    f.write(new_str)
