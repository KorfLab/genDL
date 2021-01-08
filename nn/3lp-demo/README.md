README for 3 layer perceptron demo
==================================

The code here is inspired by the following tutorial:

https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models

## Performance ##

The `run.pl` script runs the `3lp.py` program on the fabricated data
with several different network topologies. The topology of the network
didn't matter that much, so these are all 168->168->168->1.

| File1 |   MLP  |   PWM  | Notes
|:-----:|:-------|:-------|:--------------
| d.obs | 0.7905 | 0.8263 | 
| d.ex1 | 0.9188 | 0.8906 | mlp wins!
| d.ex2 | 0.8016 | 0.8243 | 
| a.obs | 0.8899 | 0.9175 | 
| a.ex3 | 0.9918 | 0.9922 | 
| a.ex4 | 0.8547 | 0.8894 | 
| a.ex5 | 0.9806 | 0.9685 | mlp wins!
| a.ex6 | 0.8234 | 0.8483 | 


