README for strawman directory
=============================

## Fabricated Data ##

+ 2021-01-05
+ Fabricated data files from repository
+ --xvalid 4 (default)
+ --seed 1

Example command line

	python3 strawman.py --file2 ../../data/acc.not.fa.gz --seed 1 --file1 ../../data/acc.ex6.fa.gz 

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



