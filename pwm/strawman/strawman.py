
import argparse

parser = argparse.ArgumentParser(
	description='Evaluate the performance of simple PWM methods')
parser.add_argument('--file1', required=True, type=str,
	metavar='<file>', help='fasta file')
parser.add_argument('--file2', required=True, type=str,
	metavar='<file>', help='fasta file')
arg = parser.parse_args()



