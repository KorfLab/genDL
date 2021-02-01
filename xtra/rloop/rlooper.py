import argparse
import json
import os

from gendl import seqio

if __name__ == '__main__':

	exe = 'rlooper' # make sure it's in your path

	parser = argparse.ArgumentParser(
		description='Wrapper for running rlooper')
	parser.add_argument('--fasta', required=True, type=str,
		metavar='<file>', help='input file in fasta format')
	parser.add_argument('--out', required=False, type=str, default='tmp',
		metavar='<file>', help='output directory [%(default)s]')
	parser.add_argument('--n', required=False, type=str, default='1500',
		metavar='<str/int>', help='size of superhelical domain [%(default)s]')
	parser.add_argument('--s', required=False, type=float, default=-0.07,
		metavar='<float>', help='superhelical density[%(default)f]')
	parser.add_argument('--a', required=False, type=float, default=10,
		metavar='<float>', help='terminal energies [%(default)f]')
	parser.add_argument('--m', required=False, type=int, default=2,
		metavar='<int>', help='minimum length [%(default)i]')
	parser.add_argument('--u', required=False, action='store_true',
		help='unconstrained')
	arg = parser.parse_args()
	# options not used: homopolymer, circular, bedfile
	# reverse, complement, invert
	# sensitivity - for the peak caller
	# residuals, dump
	# top
	# localaverageenergy
	
	# create working directory if necessary
	if not os.path.exists(arg.out): os.system(f'mkdir {arg.out}')
	
	# create fasta file with proper definition line
	names = []
	with open(f'{arg.out}/fasta', 'w') as fp:
		for name, seq in seqio.read_fasta(arg.fasta):
			tok = name.split(' ')
			names.append(tok[0])
			defline = f">{tok[0]} range=z:1:{len(seq)} 5'pad=0 3'pad=0"
			defline += ' strand=+ repeatMasking=none'
			fp.write(defline)
			fp.write('\n')
			fp.write(seq)
			fp.write('\n')
	
	# run rlooper
	cmd = (f'{exe} {arg.out}/fasta {arg.out}.output')
	cmd += f' --N {arg.n}'
	cmd += f' --sigma {arg.s}'
	cmd += f' --a arg.a'
	cmd += f' --minlength 2'
	if arg.u: cmd += f' --unconstrained'
	os.system(f'{cmd} 1>/dev/null')
	os.system(f'mv {arg.out}.output* {arg.out}')
	
	# stuff
	n = 0
	data = {}
	id = None
	with open(f'{arg.out}/{arg.out}.output_bpprob.wig') as fp:
		for line in fp.readlines():
			if line.startswith('track'): continue
			if line.startswith('browser'): continue
			if line.startswith('fixed'): continue
			if line.startswith('#'):
				id = names[n]
				n += 1
				if id not in data: data[id] = []
			else:
				data[id].append(float(line))
	
	print(json.dumps(data, indent=4))
	
	