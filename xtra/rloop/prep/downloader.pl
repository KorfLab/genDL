use strict;
use warnings;

my @url = qw(
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_INPUT_rep1_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_INPUT_rep1_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_INPUT_rep3_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_INPUT_rep3_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_INPUT_rep4_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_INPUT_rep4_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_RNaseH_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_RNaseH_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_WT_rep1_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_WT_rep1_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_WT_rep2_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_WT_rep2_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_WT_rep3_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_WT_rep3_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_WT_rep4_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_qDRIP_WT_rep4_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_sDRIP_RNaseH_LS78D_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_sDRIP_RNaseH_LS78D_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_sDRIP_WT_LS78A_rep1_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_sDRIP_WT_LS78B_rep2_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_sDRIP_WT_LS78A_rep1_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_sDRIP_WT_LS78B_rep2_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_sDRIP_WT_NV1LS1_rep1_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_sDRIP_WT_NV1LS1_rep1_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_sDRIP_WT_NV1LS2_rep2_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_sDRIP_WT_NV1LS2_rep2_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_DRIPc_WT_SX011C_rep1_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_DRIPc_WT_SX011C_rep1_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_DRIPc_WT_SX011D_rep2_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/NT2_DRIPc_WT_SX011D_rep2_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_DRIPc_WT_LS61A_rep1_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_DRIPc_WT_LS61A_rep1_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_DRIPc_WT_LS61C_rep2_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_DRIPc_WT_LS61C_rep2_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_DRIPc_WT_LS61H_rep3_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/HeLa_DRIPc_WT_LS61H_rep3_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/K562_DRIPc_WT_LS60A_rep1_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/K562_DRIPc_WT_LS60A_rep1_pos.bw
https://s3-us-west-1.amazonaws.com/muhucsc/K562_DRIPc_WT_LS60B_rep2_neg.bw
https://s3-us-west-1.amazonaws.com/muhucsc/K562_DRIPc_WT_LS60B_rep2_pos.bw
);

foreach my $url (@url) {
	my @f = split('/', $url);
	my $file = $f[4];
	if (-s "BIGWIG/$file") {
		print "skipping $file\n"
	} else {
		system("wget $url");
		system("mv $file BIGWIG")
	}
	
}
