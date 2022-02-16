nohup python3 mlp.py \
--file1 ../../data/don.hi.true.fa.gz \
--file0 ../../data/don.fake.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 --save ../save_model/don.hi.fc.pt \
> ../results/don.hi.fc.txt &

nohup python3 mlp.py \
--file1 ../../data/don.lo.true.fa.gz \
--file0 ../../data/don.fake.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 --save ../save_model/don.lo.fc.pt \
> ../results/don.lo.fc.txt &

nohup python3 mlp.py \
--file1 ../../data/don.hi.true.fa.gz \
--file0 ../../data/don.lo.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 --save ../save_model/don.hi_vs_lo.fc.pt \
> ../results/don.hi_vs_lo.fc.txt &

nohup python3 mlp.py \
--file1 ../../data/acc.hi.true.fa.gz \
--file0 ../../data/acc.fake.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 --save ../save_models/acc.hi.fc.pt \
> ../results/acc.hi.fc.txt &

nohup python3 mlp.py \
--file1 ../../data/acc.lo.true.fa.gz \
--file0 ../../data/acc.fake.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 --save ../save_models/acc.lo.fc.pt \
> ../results/acc.lo.fc.txt &

nohup python3 mlp.py \
--file1 ../../data/acc.hi.true.fa.gz \
--file0 ../../data/acc.lo.txt.fa.gz \
--layers 168 64 42 21 10 1 \
--rate 1e-2 --save ../save_models/acc.hi_vs_lo.fc.pt \
> ../results/acc.hi_vs_lo.fc.txt &
