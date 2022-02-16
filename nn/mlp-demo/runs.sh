nohup python3 mlp.py \
--file1 ../../data/don.hi.true.fa.gz \
--file0 ../../data/don.fake.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 > ../results/don.hi.fc.txt &

nohup python3 mlp.py \
--file1 ../../data/don.lo.true.fa.gz \
--file0 ../../data/don.fake.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 > ../results/don.lo.fc.txt &

nohup python3 mlp.py \
--file1 ../../data/don.hi.true.fa.gz \
--file0 ../../data/don.lo.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 > ../results/don.hi_vs_lo.fc.txt &

nohup python3 mlp.py \
--file1 ../../data/acc.hi.true.fa.gz \
--file0 ../../data/acc.fake.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 > ../results/acc.hi.fc.txt &

nohup python3 mlp.py \
--file1 ../../data/acc.lo.true.fa.gz \
--file0 ../../data/acc.fake.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 > ../results/acc.lo.fc.txt &

nohup python3 mlp.py \
--file1 ../../data/acc.hi.true.fa.gz \
--file0 ../../data/acc.lo.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 > ../results/acc.hi_vs_lo.fc.txt &
