import argparse
import yaml
import fileinput
import random
import pandas as pd
from sklearn.cluster import KMeans

from kmeanslibrary import kmeans, checking_percent_error, generating_consensus_sequence


parser = argparse.ArgumentParser(description='performing kmeans clustering')
parser.add_argument('--file', required = True, type = str,
    metavar = '<str>', help = 'yaml file for which kmeans clustering is done')
parser.add_argument('--clusters', required = False, type = int, default = 2,
    metavar = '<int>', help = 'number of clusters')
arg = parser.parse_args()



kmeans(yaml_file = arg.file, clusters = arg.clusters)
#generating_consensus_sequence(number_of_clusters = arg.clusters,
                             # data_labeled_sequences=kmeans(yaml_file=arg.file, clusters = arg.clusters))
