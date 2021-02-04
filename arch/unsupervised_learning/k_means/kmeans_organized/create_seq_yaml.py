#creating files
from kmeanslibrary import yaml_file_create_sequences
import argparse
import yaml




parser = argparse.ArgumentParser(description='Creating the yaml file that contains sequences from multiple files')
parser.add_argument('--file1', required = True, type = str,
    metavar = '<str>', help = 'file used yaml file')
parser.add_argument('--file2', required = False, type = str,
    metavar = '<str>', help = 'file used yaml file')
parser.add_argument('--file3', required = False, type = str,
    metavar = '<str>', help = 'file used yaml file')
parser.add_argument('--num', required=False, default = 10, type = int,
    metavar = '<int>', help = 'number of sequences we grab from each file')
parser.add_argument('--name', required = False, default = 'mixed_sequences',
    metavar = '<str>', help = 'naming the yaml file used for kmeans_clustering')
parser.add_argument('--min', required=False, type = int, default = 0,
    metavar = '<int>', help = 'start point')
parser.add_argument('--max', required=False, type = int, default = 42,
    metavar = '<int>', help = 'end point')

arg = parser.parse_args()


yaml_file_create_sequences(file1 = arg.file1, file2 = arg.file2, file3 = arg.file3,
                           minimum=arg.min, maximum=arg.max, name=arg.name, num = arg.num)

