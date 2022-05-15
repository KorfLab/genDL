
my @file0 = qw(dist rnda rndd rndp);
foreach my $f (@file0) {
	system("python3 ../../nn/mlp-demo/mlp.py --file1 prox.fa.gz --file0 $f.fa.gz --layers 200 100 50 1");
}
