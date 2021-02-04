import fileinput
import os
import random
import argparse
#from Bio.Nexus import Nexus
import subprocess


#parser
parser = argparse.ArgumentParser(description='Making it more neat')
parser.add_argument('--file', required = True, type = str,
    metavar = '<str>', help = 'file used the nexus file is created')
parser.add_argument('--num', required=False, type = int, default = 10,
    metavar = '<int>', help = 'number of string for phylogenetic tree')
parser.add_argument('--min', required=False, type = int, default = 0,
    metavar = '<int>', help = 'start point')
parser.add_argument('--max', required=False, type = int, default = 42,
    metavar = '<int>', help = 'end point')
parser.add_argument('--name', required=False, type = str, default = 'compare',
    metavar = '<str>', help = 'naming nexus file')
parser.add_argument('--multiple', required = False, type = str, default = 'n',
    metavar = '<str>', help = 'y/n for the multiple phylogenetic trees for the same file')
arg = parser.parse_args()


#edit distance function
def edit(d1, d2):
    d = 0
    for i in range(len(d1)):
        if d1[i] != d2[i]:
            d += 1
    return d

#read the sequences and append them to the sequences list
sequences= []
#print(type(arg.file))
if arg.multiple == 'n':
    with open((arg.file),"r") as file:
        line = file.read().splitlines()
        print(line)
        random.shuffle(line)
        print(line)
    #print(line)
        for i in range(arg.num):
            sequences.append(line[i][arg.min:arg.max])
elif arg.multiple == 'y':
    for line in fileinput.input(arg.file):
        line = line.rstrip() # remove newline (return character), often useful
        sequences.append(str(line[arg.min:arg.max])) # store the data

print((sequences), len(sequences))


#create 2D table for distances
table = []
for i in range(len(sequences)):
    table.append([])
#print(sequences)
#load up the table
for s1 in range(len(sequences)):
    for sn in range(len(sequences)):
        #print(sequences[s1], sequences[sn])
        if sequences[s1] == sequences[sn]:
            table[sn].append(0)
        else:
            dist = edit(sequences[s1], sequences[sn])
            table[sn].append(dist)

#print(table)
##notes:
##mention the number of sequences and the shuffle at least one (there is a relation between)
#print(table, len(table))
my_file = open(arg.name + '.nex', 'w')
my_file.write('#NEXUS' + '\n')
my_file.write('BEGIN Taxa;' + '\n')
my_file.write('DIMENSIONS NTax=' + str(len(sequences)) + ';' + '\n')
my_file.write('TAXLABELS' + '\n')

for i in range(len(table)):
    my_file.write(f'[{i}]' + ' ' + f's{i}' + '\n')

my_file.write(';' + '\n')
my_file.write('END;' + '\n')

my_file.write('BEGIN distances;' + '\n')
my_file.write('format triangle = upper;' + '\n')
my_file.write('matrix' + '\n')

#print(table, len(table))
for i in range(len(table)):
    row = []
    my_file.write(f's{i}')
    (row.append('\t'*i))

    for j in range(i, len(table)):

        row.append(str(table[i][j]))


    my_file.write('\t'.join(row) + '\n')
my_file.write(';' + '\n')
my_file.write('END;' + '\n')


file_name = str(arg.name)+'.nex'
file_name1 = str(arg.name)+'.svg'

#os.system('xvfb-run --auto-servernum --server-num=1 SplitsTree -g')

#ipc - interprocess communication

'''

command file, input file

command file
begin Splitstree
LOAD FILE=/Users/Amalia/Tutorials1/NNE/genDL/unsupervised_learning/phylogenetic_tree/my_file
UPDATE
EXPORTGRAPHICS format=SVG TEXTASSHAPES=YES file=/Users/Amalia/Tutorials1/NNE/genDL/unsupervised_learning/phylogenetic_tree/file_name1
QUIT
end;

global tree?
donor vs accepto

my_file.write(';' + '\n')
my_file.write('END;' + '\n')

os.system("""osascript -e 'tell app "SplitsTree" to open'""")
fileName = listbox_1.get(ACTIVE)
os.system("notepad.exe " + fileName)


path_to_splitstree = '/Volumes/Macintosh_HD/Applications/SplitsTree/SplitsTree.app'
path_to_file = '/Users/Amalia/Tutorials1/NNE/genDL/unsupervised_learning/phylogenetic_tree/'+str(arg.name)+'.nex'
subprocess.Popen(['sudo', ])
subprocess.Popen([path_to_splitstree, path_to_file])
#output, error = process.communicate()


#SplitsTreeCMD
command = ['chmod u+x /Volumes/Macintosh_HD/Applications/SplitsTree/SplitsTreeCMD.app/',
            '/Users/Amalia/Tutorials1/NNE/genDL/unsupervised_learning/phylogenetic_tree/'+str(arg.name)+'.nex']
#path_to_file = '/Users/Amalia/Tutorials1/NNE/genDL/unsupervised_learning/phylogenetic_tree/'+str(arg.name)+'.nex'
process = subprocess.Popen([command[0].split(), command[1]], stdout=subprocess.PIPE)
output, error = process.communicate()
'''
