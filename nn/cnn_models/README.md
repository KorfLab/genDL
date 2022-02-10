README for CNN Models
==================================

The code here is inspired by the following tutorial:

https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models

## Performance ##

The `run.pl` script runs the `3lp.py` program on the fabricated data
with several different network topologies. The topology of the network
didn't matter that much, so these are all 168->168->168->1.

| File1            |  File0  |  CNN   |  MLP   |   PWM  | Notes
|:----------------:|:-------:|:------:|:------:|:------:|
| d.obs            | don.not | 0.825  | 0.8344 | 0.8263 |
| d.obs.rearranged | don.not | 0.958  | 0.9725 |        |
| d.ex1            | don.not | 0.935  | 0.9188 | 0.8906 |
| d.ex1.rearranged | don.not | 0.9700 | 0.9700 |        |
| d.ex2            | don.not | 0.825  | 0.8303 | 0.8243 |
| d.ex2.rearranged | don.not | 0.963  | 0.9695 |        |
| a.obs            | acc.not | 0.9100 | 0.9144 | 0.9175 |
| a.obs.rearranged | acc.not | 0.984  | 0.9795 |        |
| a.ex3            | acc.not | 0.9920 | 0.9918 | 0.9922 |
| a.ex3.rearranged |acc.not	 | 0.9860	| 0.9800 |        |
| a.ex4            | acc.not | 0.8950 | 0.8942 | 0.8894 |
| a.ex4.rearranged | acc.not | 0.976	| 0.9817 |        |
| a.ex5            | acc.not | 0.9845 | 0.9806 | 0.9685 |
| a.ex5.rearranged | acc.not | 0.9865 | 0.9797 |        |
| a.ex6            | acc.not | 0.8480 | 0.8514 | 0.8483 |
| a.ex6.rearranged | acc.not | 0.9741	| 0.9777 |        |

Observations
+ CNN and FFN have comparable performance
+ Adding more convolutions or fully connected layers does not correlate to better performance


New Data
+ Splice sites with motifs anchored in place
+ Splice sites with motifs shifted around
+ TIS motifs (ATG) shifted around
