#multiple sequences
import argparse
import random

parser = argparse.ArgumentParser(description='Creating the file with sequences from multiple files')
parser.add_argument('--file1', required = True, type = str,
    metavar = '<str>', help = 'file used the nexus file is created')
parser.add_argument('--file2', required = False, type = str,
    metavar = '<str>', help = 'file used the nexus file is created')
parser.add_argument('--file3', required = False, type = str,
    metavar = '<str>', help = 'file used the nexus file is created')
parser.add_argument('--num', required=False, default = 50, type = int,
    metavar = '<int>', help = 'number of sequences we grab from each file')
parser.add_argument('--name', required = False, default = 'mixed_sequences',
    metavar = '<str>', help = 'naming the txt file used for phylo_tree')
arg = parser.parse_args()

'''
data = []
for line in fileinput.input():
    #if line[0] == '#': continue # skip over comments
    if line.startswith('#'): continue # same as above
    line = line.rstrip() # remove newline (return character), often useful
    data.append(float(line)) # store the data
'''
sequences= []
with open((arg.file1),"r") as file:
    line = file.read().splitlines()
    for i in range(arg.num):
        choosing_seqs = random.choice(line)
        sequences.append(choosing_seqs)
print(len(sequences))

if arg.file2 is not None:
    with open((arg.file2),"r") as file:
        line = file.read().splitlines()
        #print(len(line))
        for i in range(arg.num):
            choosing_seqs = random.choice(line)
            sequences.append(choosing_seqs)
print(len(sequences))

if arg.file3 is not None:
    with open((arg.file3),"r") as file:
        line = file.read().splitlines()
        #print(len(line))
        for i in range(arg.num):
            choosing_seqs = random.choice(line)
            sequences.append(choosing_seqs)
print(len(sequences),sequences)

f= open(arg.name+'.txt','w+')

f.write('\n'.join(map(str, sequences)))
f.close()
