#python3 mlp.py --file1 ../../data42/at.don.fa.gz --file0 ../../data42/at.n2don.fa.gz --layers 168 1

my @gen = qw(at ce dm);
my @sig = qw(don acc);
my @neg = qw(n1 n2 n3);

foreach my $gen (@gen) {
	foreach my $sig (@sig) {
		foreach my $neg (@neg) {
			print STDERR "$gen $sig $neg\n";
			system("python3 mlp.py --file1 ../../data42/$gen.$sig.fa.gz --file0 ../../data42/$gen.$neg$sig.fa.gz --layers 168 1");
		}
	}
}
