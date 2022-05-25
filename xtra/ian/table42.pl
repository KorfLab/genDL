use strict;
use warnings;

my @prog = qw(pwm wam1 wam2 wam3 per mlp1 mlp2);
my %exec = (
	'pwm' => "python3 ../../pwm/strawman/strawman.py",
	'wam1' => "python3 ../../pwm/wam/wam.py --order 1",
	'wam2' => "python3 ../../pwm/wam/wam.py --order 2",
	'wam3' => "python3 ../../pwm/wam/wam.py --order 3",
	'per'  => "python3 ../../nn/mlp-demo/mlp.py --layers 168 1",
	'mlp1' => "python3 ../../nn/mlp-demo/mlp.py --layers 168 84 1",
	'mlp2' => "python3 ../../nn/mlp-demo/mlp.py --layers 168 84 42 1",
	'mlp3' => "python3 ../../nn/mlp-demo/mlp.py --layers 168 84 42 21 1",
);
my $ddir = "../../data/splice42";
my @neg = qw(n1 n2 n3 n4);
my @genome = qw(at ce dm);
my @site = qw(don acc);
my @limit = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384);
my $runs = 10;

foreach my $gen (@genome) {
	foreach my $prog (@prog) {
		foreach my $site (@site) {
			foreach my $neg (@neg) {
				foreach my $limit (@limit) {
					my $total = 0;
					for (my $i = 0; $i < $runs; $i++) {
						my $f1 = "--file1 $ddir/$gen.$site.fa.gz";
						my $f0 = "--file0 $ddir/$gen.$neg$site.fa.gz";
						my $n = "--limit $limit";
						my $cli = "$exec{$prog} $f1 $f0 $n\n";
						my $acc = `$cli`;
						$total += $acc;
					}
					my $ave = $total / $runs;
					print join("\t", $gen, $prog, $site, $limit, $neg, $ave), "\n";
				}
			}
		}
	}
}
