###Notes
#Figure out how to add labels
#Iteration
#Validation Data
#Search more on starting points

#centroids

#pickled dictionary

###
#from matplotlib import pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.decomposition import PCA
###

import numpy as np
import fileinput
import yaml
import pprint
from collections import OrderedDict



from numpy import array
import argparse
import random
import pandas as pd


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description='Making it more neat')
parser.add_argument('--file', required = True, type = str,
    metavar = '<str>', help = 'file used the nexus file is created')
parser.add_argument('--num', required=False, type = int, default = 10,
    metavar = '<int>', help = 'number of string for phylogenetic tree')
parser.add_argument('--min', required=False, type = int, default = 0,
    metavar = '<int>', help = 'start point')
parser.add_argument('--max', required=False, type = int, default = 42,
    metavar = '<int>', help = 'end point')
parser.add_argument('--clusters', required = False, type = int, default = 2,
    metavar = '<int>', help = 'number of clusters')
parser.add_argument('--multiple', required = False, type = str, default = 'n',
    metavar = '<str>', help = 'y/n for the multiple phylogenetic trees for the same file')
#parser.add_argument('--number_of_files', required = False, type = int, default = 1,
    #metavar = '<int>', help = 'how many files were used when creating a mixed_sequence file')
arg = parser.parse_args()

def edit(d1, d2):
    d = 0
    for i in range(len(d1)):
        if d1[i] != d2[i]:
            d += 1
    return d

#read the sequences

sequences= []
keys = []

import sys

if arg.multiple == 'n':
    with open((arg.file),"r") as file:
        line = file.read().splitlines()
        random.shuffle(line)
        #print(len(line))
        #sys.exit()
        for i in range(arg.num):
            sequences.append(line[i][arg.min:arg.max])

elif arg.multiple == 'y':
    with open(arg.file) as f:
        information = yaml.load_all(f, Loader=yaml.FullLoader)
        print(type(information))
        #information.keys()
        for i in information:
            #dictionary = i
            #print(dictionary)
            counter = 0 ###
            for k, s in i.items():
                keys.append(k)
                sequences.append([]) ###

                for substring in (s):
                    sequences[counter].append(substring[arg.min:arg.max]) ###

                counter += 1

#print(sequences, len(sequences[0]), len(sequences[1]))


complete_table = []
for i in sequences:
    complete_table += i

assert(arg.clusters <= len(complete_table))

list_bases = {'A': 1.0, 'C': 2.0, 'G': 3.0, 'T': 4.0}


table = []
for i in range(len(complete_table)):
    table.append([])

for item in range(len(complete_table)):
    #print(item)
    for i in (complete_table[item]):
        #print(item[i])
        if i in list_bases.keys():
            #print(i)
            table[item].append(list_bases[i])




#print(len(table), len(table[0]), len(table[1]))
#print(table)





df = pd.DataFrame(table)
#df = df.transpose()
#print(df)

headers = []
for i in range(len(sequences[0][0])):
    headers.append(str(f'p{i}'))
#print(headers)

df.columns = headers


#print(df)

kmeans = KMeans(n_clusters=arg.clusters).fit(df)
#decreasing tolerance, increasing toleration
centroids = kmeans.cluster_centers_
#print('Centroids:', centroids)
#print('Kmeans:', kmeans)
#print('labels:', kmeans.labels_)
#print(type(kmeans.labels_))



if arg.multiple == 'y':
    #print(arg.num)
    array_div = len(table)//len(keys)
    #print('цифра:', array_div)

elif arg.multiple == 'n':
    pass

nested_dist = {}
#summary_dict = {}

for i in range(len(keys)):
    nested_dist[keys[i]] = {}
    analyzing = kmeans.labels_[i*array_div:((i+1)*array_div)]
    #print(type(analyzing))

    #print(analyzing, len(analyzing))
    for checking_label in analyzing:
        if checking_label in nested_dist[keys[i]]:
            nested_dist[keys[i]][checking_label] += 1
        else:
            nested_dist[keys[i]][checking_label] = 1


#print(nested_dist)
neat_dict = pprint.pformat(nested_dist)
#print(neat_dict)

#nested_dist = OrderedDict(sorted(nested_dist.items()))
#clusters
#information.keys

for key, value in nested_dist.items():
    print(key, '--')
    # Again iterate over the nested dictionary
    for label, percentage in value.items():
        print(label, ':', percentage, '\t', '{:.3f}'.format(percentage/array_div))



'''




if arg.multiple == 'n':
    with open((arg.file),"r") as file:
        line = file.read().splitlines()
        random.shuffle(line)
        #print(line)
        for i in range(arg.num):
            sequences.append(line[i][arg.min:arg.max])
elif arg.multiple == 'y':
    for line in fileinput.input(arg.file):
        line = line.rstrip() # remove newline (return character), often useful
        sequences.append(str(line[arg.min:arg.max])) # store the data

bases = {'A': 1.0, 'C': 2.0, 'G': 3.0, 'T': 4.0}

table = []
for i in range(len(sequences)):
    table.append([])

for item in range(len(sequences)):
    #print(item)
    for i in (sequences[item]):
        #print(item[i])
        if i in bases.keys():
            #print(i)
            table[item].append(bases[i])

print(table, '\n', len(table), '\n', len(table[0]))


#table_reshape = np.array(table).reshape((arg.num,len(table[0])))
#print(table_reshape, len(table_reshape[0]))
#table_map = array(table)
df = pd.DataFrame(table)
#df = df.transpose()


headers = []
for i in range(len(table[0])):
    headers.append(str(f'p{i}'))
df.columns = headers
#print(df)



###elbow method


pca = PCA(2)
pca.fit(df)

pca_data = pd.DataFrame(pca.transform(df))

#print(pca_data.head())


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import math



colors = list(zip(*sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                           for name, color in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).items())))[1]

skips = math.floor(len(colors[5 : -5])/arg.clusters)
cluster_colors = colors[5 : -5 : skips]

#fig = plt.figure()
#fig, ax = plt.subplots()
#print(kmeans.labels_)
plt.scatter(pca_data[0], pca_data[1], c = kmeans.labels_)


#for i, txt in enumerate(variablelabel):
    #variable.annotate(txt, (test1[i], test2[i]))

for i in range(len(table[0])):
    ax.text(pca_data[0], pca_data[1], '%s' % (str(i)))



#for i in range(0,arg.num):

#    plt.text(arg.num[i,0],arg.num[i,1], '%s' % (str(i)), size=20, zorder=1,  color='k')

plt.show()


#import seaborn as sns
# generating correlation heatmap
#sns.heatmap(zoo_data.corr(), annot = True)

# posting correlation heatmap to output console
#plt.show()


from matplotlib import colors as mcolors
import math

# Generating different colors in ascending order of their hsv values
colors = list(zip(*sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
for name, color in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).items())))[1]


# number of steps to taken generate n(clusters) colors
skips = math.floor(len(colors[5 : -5])/clusters)
cluster_colors = colors[5 : -5 : skips]

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pca_data[0], pca_data[1], pca_data[2],
           c = list(map(lambda label : cluster_colors[label],
                                            kmeans.labels_)))

str_labels = list(map(lambda label:'% s' % label, kmeans.labels_))

list(map(lambda data1, data2, data3, str_label:
        ax.text(data1, data2, data3, s = str_label, size = 16.5,
        zorder = 20, color = 'k'), pca_data[0], pca_data[1],
        pca_data[2], str_labels))

plt.show()


#plt.scatter(df['s0'], df['s1'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
#plt.show()


1. shuffle (done)
2. mixing data (done)
3. statistical measurement of phylogenetic trees = research the topic
4. kmeans acgt = numbers
'''

