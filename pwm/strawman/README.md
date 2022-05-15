README for strawman directory
=============================

## Fabricated Data ##

+ 2021-01-05
+ Fabricated data files from repository
+ --xvalid 4 (default)
+ --seed 1

Example command line

	python3 strawman.py --file0 ../../data/acc.not.fa.gz --seed 1 --file1 ../../data/acc.ex6.fa.gz

Results

| file1   | file0   | accuracy | pwm info| notes
|:--------|:--------|:---------|:--------|:--------
| don.obs | don.not |  0.8263  |  6.2386 |
| don.ex1 | don.not |  0.8906  |  7.1415 |
| don.ex2 | don.not |  0.8243  |  6.2073 |
| acc.obs | acc.not |  0.9175  |  8.5820 |
| acc.ex3 | acc.not |  0.9922  | 10.0112 | too easy
| acc.ex4 | acc.not |  0.8894  |  7.3852 |
| acc.ex5 | acc.not |  0.9685  |  8.9148 |
| acc.ex6 | acc.not |  0.8483  |  6.8941 |

## Real Data ##

From `datacore/project_gendl`

+ n1-3: negative models 1-3, see datacore
+ pwm: strawman.py
+ wam1: wam.py --order 1
+ wam2: wam.py --order 2
+ wam3: wam.py --order 3
+ per: mlp.py --layers 168 1
+ 3lp: 3lp.py --layer2 84 --layer3 42

| spe | f1  | f0 | pwm  | wam1 | wam2 | wam3 | per  | 3lp  |
|:---:|:---:|:--:|:----:|:----:|:----:|:----:|:----:|:----:|
| at  | don | n1 | 94.5 | 95.6 | 95.9 | 96.0 | 94.9 | 94.7 |
| at  | don | n2 | 89.4 | 91.7 | 92.2 | 92.1 | 89.6 | 89.6 |
| at  | don | n3 | 49.4 | 72.7 | 75.1 | 75.7 | 50.0 | 68.3 |
| at  | acc | n1 | 93.6 | 95.5 | 95.8 | 95.6 | 93.9 | 93.7 |
| at  | acc | n2 | 80.1 | 86.5 | 87.7 | 87.7 | 80.2 | 80.5 |
| at  | acc | n3 | 49.1 | 75.9 | 78.3 | 78.3 | 50.0 | 68.9 |
| ce  | don | n1 | 94.2 | 95.6 | 96.0 | 96.0 | 94.7 | 94.6 |
| ce  | don | n2 | 86.7 | 90.8 | 91.6 | 91.4 | 87.7 | 88.2 |
| ce  | don | n3 | 49.3 | 75.1 | 78.0 | 78.0 | 50.3 | 70.4 |
| ce  | acc | n1 | 97.5 | 97.9 | 98.0 | 97.9 | 97.5 | 97.2 |
| ce  | acc | n2 | 89.6 | 91.7 | 92.4 | 92.4 | 89.7 | 88.7 |
| ce  | acc | n3 | 49.1 | 72.3 | 75.3 | 75.4 | 49.8 | 65.4 |
| dm  | don | n1 | 92.7 | 94.0 | 94.2 | 93.4 | 93.1 | 92.1 |
| dm  | don | n2 | 89.8 | 91.5 | 91.8 | 91.0 | 90.6 | 89.1 |
| dm  | don | n3 | 49.2 | 66.7 | 70.0 | 68.9 | 49.8 | 62.3 |
| dm  | acc | n1 | 91.1 | 93.0 | 93.2 | 92.2 | 91.3 | 90.2 |
| dm  | acc | n2 | 76.1 | 81.9 | 82.7 | 81.4 | 76.0 | 75.4 |
| dm  | acc | n3 | 48.7 | 71.2 | 73.4 | 71.8 | 49.3 | 61.9 |
