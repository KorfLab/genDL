#multiple sequences
import argparse
import random
import yaml

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
parser.add_argument('--min', required=False, type = int, default = 0,
    metavar = '<int>', help = 'start point')
parser.add_argument('--max', required=False, type = int, default = 42,
    metavar = '<int>', help = 'end point')
arg = parser.parse_args()

'''
data = []
for line in fileinput.input():
    #if line[0] == '#': continue # skip over comments
    if line.startswith('#'): continue # same as above
    line = line.rstrip() # remove newline (return character), often useful
    data.append(float(line)) # store the data
'''
dict_k_means = {}

value1 = []
with open((arg.file1),"r") as file:
    key1 = arg.file1.replace(".txt","")

    line = file.read().splitlines()
    random.shuffle(line)
    for i in range(arg.num):
        print(line[i][arg.min:arg.max])
        value1.append(line[i])

dict_k_means[key1] = value1
print(len(value1))



if arg.file2 is not None:
    value2 = []
    with open((arg.file2),"r") as file:
        key2 = arg.file2.replace(".txt","")

        line = file.read().splitlines()
        random.shuffle(line)
        #print(len(line))
        for i in range(arg.num):
            value2.append(line[i][arg.min:arg.max])

    dict_k_means[key2] = value2
    print(len(value2))



if arg.file3 is not None:
    value3 = []
    with open((arg.file3),"r") as file:
        key3 = arg.file3.replace(".txt","")

        line = file.read().splitlines()
        random.shuffle(line)
        #print(len(line))
        for i in range(arg.num):
            value3.append(line[i][arg.min:arg.max])

    dict_k_means[key3] = value3

    print(len(value3))
#print(dict_k_means)

new = yaml.dump(dict_k_means)
#print(new)

#f= open(arg.name+'.txt','w+')

with open(arg.name+'.yaml', 'w') as f:
    data = yaml.dump(dict_k_means, f)

#f.write('\n'.join(map(str, sequences)))
#f.close()
