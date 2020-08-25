###
import numpy as np
#import fileinput
from matplotlib import pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.cluster import KMeans

from Bio.Cluster import kcluster
from Bio.Cluster import kmedoids
from Bio.Cluster import somcluster
from numpy import array
import argparse
import random
import seaborn as sns

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
arg = parser.parse_args()

def edit(d1, d2):
    d = 0
    for i in range(len(d1)):
        if d1[i] != d2[i]:
            d += 1
    return d

#read the sequences

sequences= []
'''
for line in fileinput.input():
    line = line.rstrip()
    sequences.append((line))
'''
with open((arg.file),"r") as file:
    line = file.read().splitlines()
    for i in range(arg.num):
        choosing_seqs = random.choice(line)
        sequences.append(choosing_seqs[arg.min:arg.max])
print(len(sequences))
bases = {'A': 1, 'C': 2, 'T': 3, 'G': 4}

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


#print(table, len(table), len(table[0]))


table_map = array(table)
#print(table_map)

#clusterid, error, found = kmedoids(table)
#clusterid, map = somcluster(table_map)
#print('clustered', clusterid)
#print('map', map)

#bio python
#kmeans
clusterid, error, found = kcluster(table_map, arg.clusters)
print(clusterid, len(clusterid))
#print(found)
#cluster_id = clusterid.tolist()

#xyz=np.array(np.random.random((100)))
#print(xyz, type(xyz))
#plt.scatter(clusterid[0][:,0], clusterid[0][:,1], c=data[1], cmap='viridis')
#print(clusterid)
#plt.scatter(xyz[:,0], xyz[:,1])
#plt.show()
#plt.scatter(clusterid, aspect = 'auto')

clusterid_, map = somcluster(table_map)
#print(clusterid_)
#print(map)
#SOMS


 #center of the cluster



'''
#sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Scaling the data to normalize
model = KMeans(n_clusters=5).fit(X)

# Visualize it:
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
'''


'''
### creates 2D dataframe
df = pd.DataFrame(table)
df = df.transpose()

headers = []
for i in range(len(sequences)):
    headers.append(str(f's{i}'))
df.columns = headers


### A - number,



from matplotlib import colors as mcolors
import math


colors = list(zip(*sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
for name, color in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).items())))[1]


# number of steps to taken generate n(clusters) colors
skips = math.floor(len(colors[5 : -5])/clusters)
cluster_colors = colors[5 : -5 : skips]

##clustering
Kmean = KMeans(n_clusters=2)
Kmean.fit(table)
KMeans(algorithm=’auto’, copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
 random_state=None, tol=0.0001, verbose=0)

print(df)
print()

1. shuffle (done)
2. mixing data (done)
3. statistical measurement of phylogenetic trees = research the topic
4. kmeans acgt = numbers
'''

