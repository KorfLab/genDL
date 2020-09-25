import argparse
import random
import gzip

parser = argparse.ArgumentParser(description='Creating the file with sequences from multiple files')
parser.add_argument('--file1', required = True, type = str,
    metavar = '<str>', help = 'file used the txt.gz file is created')
parser.add_argument('--file2', required = False, type = str,
    metavar = '<str>', help = 'file used the txt.gz file is created')
parser.add_argument('--file3', required = False, type = str,
    metavar = '<str>', help = 'file used the txt.gz file is created')
parser.add_argument('--num', required=False, default = 1000, type = int,
    metavar = '<int>', help = 'number of sequences we grab from each file')
parser.add_argument('--name', required = False, default = 'mixed_sequences',
    metavar = '<str>', help = 'naming the txt file used for phylo_tree')
parser.add_argument('--min', required=False, type = int, default = 0,
    metavar = '<int>', help = 'start point')
parser.add_argument('--max', required=False, type = int, default = 42,
    metavar = '<int>', help = 'end point')
arg = parser.parse_args()


def writing_to_txt_file(file, limit, start, end):
    with gzip.open(file, mode='rt') as fp:
            lines = fp.read().splitlines()
            if limit > len(lines):
                sys.stderr.write(f'limit {limit} exceeds sequences {len(lines)}\n')
            random.shuffle(lines)
            for i in range(limit):
                f.write(lines[i][start:end]+'\n')
    return


f = open(arg.name+'.txt', 'a')

writing_to_txt_file(arg.file1, arg.num, arg.min, arg.max)

if arg.file2 is not None:
    writing_to_txt_file(arg.file2, arg.num, arg.min, arg.max)

if arg.file3 is not None:
    writing_to_txt_file(arg.file3, arg.num, arg.min, arg.max)


f.close()



