README for wam directory
========================

What the heck is a WAM? It's a PWM with n-th order Markov model context.
In other words, a PWM is a 0th-order WAM. In other, other words, each
position of a WAM specifies the probability of emitting an A, C, G, or T
given some previous context, such as AA. So the probability of emitting
AAA might be very different than CCA because the context, here AA or CC,
may be very different.

## Fabricated Data ##

+ 2021-01-28
+ Fabricated data files from repository
+ --xvalid 4 (default)
+ --seed 1

Example command line

	python3 wam.py --file1 ../../data/don.obs.fa.gz --file0 ../../data/don.not.fa.gz  --seed 1 --order 1

Results

| file1   | file0   | mm | accuracy | strawman | notes
|:--------|:--------|:--:|----------|:---------|:-----
| don.obs | don.not |  1 | 0.8212   | 0.8263   | 
| don.ex1 | don.not |  1 | 0.9323   | 0.8906   | x
| don.ex2 | don.not |  1 | 0.8267   | 0.8243   | 
| acc.obs | acc.not |  1 | 0.9158   | 0.9175   | 
| acc.ex3 | acc.not |  1 | 0.9921   | 0.9922   | 
| acc.ex4 | acc.not |  1 | 0.8890   | 0.8894   | 
| acc.ex5 | acc.not |  1 | 0.9685   | 0.9685   | 
| acc.ex6 | acc.not |  1 | 0.8505   | 0.8483   | 
| don.obs | don.not |  2 | 0.8146   | 0.8263   | 
| don.ex1 | don.not |  2 | 0.9299   | 0.8906   | x
| don.ex2 | don.not |  2 | 0.8269   | 0.8243   | 
| acc.obs | acc.not |  2 | 0.9079   | 0.9175   | 
| acc.ex3 | acc.not |  2 | 0.9921   | 0.9922   | 
| acc.ex4 | acc.not |  2 | 0.8770   | 0.8894   | 
| acc.ex5 | acc.not |  2 | 0.9855   | 0.9685   | x
| acc.ex6 | acc.not |  2 | 0.8503   | 0.8483   | 
| don.obs | don.not |  3 | 0.7921   | 0.8263   | 
| don.ex1 | don.not |  3 | 0.9249   | 0.8906   | x
| don.ex2 | don.not |  3 | 0.8050   | 0.8243   | 
| acc.obs | acc.not |  3 | 0.8902   | 0.9175   | 
| acc.ex3 | acc.not |  3 | 0.9964   | 0.9922   | 
| acc.ex4 | acc.not |  3 | 0.8631   | 0.8894   | 
| acc.ex5 | acc.not |  3 | 0.9837   | 0.9685   | x
| acc.ex6 | acc.not |  3 | 0.8319   | 0.8483   | 


