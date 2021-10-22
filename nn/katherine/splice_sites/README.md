README for 3 layer MLP on splice site data
==================================

## Performance ##

Topology: 84 -> 15 -> 1

| File1 |   File0   |  MLP   |  PWM   | Epochs | Notes
|:-----:|:----------|:-------|:-------|:-------|:--------------
| d.obs | don.false | 0.7905 | 0.8263 |
| d.ex1 | don.false | 0.9188 | 0.8906 |        | mlp wins!
| d.ex2 | don.false | 0.8016 | 0.8243 |  800   |
| a.obs | acc.false | 0.8899 | 0.9175 |  800   |
| a.ex3 | acc.false | 0.9928 | 0.9922 |  800   | mlp wins with poss overfitting
| a.ex4 | acc.false | 0.8648 | 0.8894 |  800   | poss overfitting
| a.ex5 | acc.false | 0.9753 | 0.9685 |  800   | mlp wins!
| a.ex6 | acc.false | 0.8426 | 0.8483 |  800   | slightly overfitted
