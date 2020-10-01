#!/usr/bin/env python3

import argparse
import strawlib

parser = argparse.ArgumentParser(
	description='PWM-based discriminator.')

parser.add_argument('--true', required=True, type=str,
	metavar='<path>', help='.gz file of true sequences')
parser.add_argument('--fake', required=True, type=str,
	metavar='<path>', help='.gz file of fake sequences')
parser.add_argument('--nt', required=False, type=int, default=1000,
	metavar='<int>', help='number of true sequences to use [%(default)i]')
parser.add_argument('--nf', required=False, type=int, default=1000,
	metavar='<int>', help='number of fake sequences to use [%(default)i]')
parser.add_argument('--x', required=False, type=int, default=2,
	metavar='<int>', help='cross-validation level [%(default)i]')
parser.add_argument('--kt', required = False, type = int, default = 2,
	metavar='<int>', help='number of clusters for trues')
parser.add_argument('--kf', required = False, type = int, default = 2,
	metavar='<int>', help='number of clusters for fakes')
parser.add_argument('--start', required = False, type = int, default = 0,
	metavar='<int>', help='start of seqeunce')
parser.add_argument('--end', required = False, type = int, default = 42,
	metavar='<int>', help='end of seqeunce')
parser.add_argument('--regex', action='store_true', help='test regex')
parser.add_argument('--pwm', action='store_true', help='test pwm w/ threshold')
parser.add_argument('--pvp', action='store_true', help='test pwm vs pwm')
parser.add_argument('--boost', action='store_true', help='test boosting pwm')
parser.add_argument('--kmer', action='store_true', help='test kmers')
parser.add_argument('--kpwm', action='store_true', help='test k pwms')
arg = parser.parse_args()

trues = strawlib.get_seqs(arg.true, arg.nt, arg.start, arg.end)
fakes = strawlib.get_seqs(arg.fake, arg.nf, arg.start, arg.end)
print('true_sequences:', len(trues))
print('fake_sequences', len(fakes))

if arg.regex:
	acc = strawlib.regex(trues, fakes, arg.x)
	print(f'REGEX: {acc:.4f}')

if arg.pwm:
	acc = strawlib.pwm_threshold(trues, fakes, arg.x)
	print(f'PWM Threshold: {acc:.4f}')

if arg.pvp:
	acc = strawlib.pwm_vs_pwm(trues, fakes, arg.x)
	print(f'PWM vs. PWM: {acc:.4f}')

if arg.boost:
	acc = strawlib.boosted_pwms(trues, fakes, arg.x)
	print(f'Boosted PWMs: {acc:.4f}')

if arg.kmer:
	acc = strawlib.kmer_threshold(trues, fakes, arg.x)
	print(f'KMER Threshold: {acc:.4f}')

if arg.kpwm:
	acc = strawlib.kmeans_pwm(trues, fakes, arg.kt, arg.kf, arg.x)
	print(f'Kmeans PWMs: {acc:.4f}')
