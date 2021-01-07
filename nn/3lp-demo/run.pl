#!/usr/bin/perl
use strict;
use warnings;

# network node sizes
my @lay = qw(336 168 84 42);

# donor experiments
my $file2 = 'don.not';
my @file1 = qw(don.obs don.ex1 don.ex2);

foreach my $file1 (@file1) {
	foreach my $l2 (@lay) {
		foreach my $l3 (@lay) {
			my $cmd = "python3 3lp.py --file1 ../../data/$file1.fa.gz --file2 ../../data/$file2.fa.gz --seed 1 --layer2 $l2 --layer3 $l3";
			print STDERR "$cmd\n";
			system("python3 3lp.py --file1 ../../data/$file1.fa.gz --file2 ../../data/$file2.fa.gz --seed 1 --layer2 $l2 --layer3 $l3") == 0 or die;
		}
	}
}

# acceptor experiments
$file2 = 'acc.not';
@file1 = qw(acc.obs acc.ex3 acc.ex4 acc.ex5 acc.ex6);

foreach my $file1 (@file1) {
	foreach my $l2 (@lay) {
		foreach my $l3 (@lay) {
			my $cmd = "python3 3lp.py --file1 ../../data/$file1.fa.gz --file2 ../../data/$file2.fa.gz --seed 1 --layer2 $l2 --layer3 $l3";
			print STDERR "$cmd\n";
			system("python3 3lp.py --file1 ../../data/$file1.fa.gz --file2 ../../data/$file2.fa.gz --seed 1 --layer2 $l2 --layer3 $l3") == 0 or die;
		}
	}
}
