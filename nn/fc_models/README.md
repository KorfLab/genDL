# Results

## C. elegans, motif in different locations
| splice site | positive | negative | hidden layers | epochs | testing accuracy |
|:-----------:|:--------:|:--------:|:-------------:|:------:|:--------:|
| acc         |  ce.acc  | acc.n1   |       1       |   20   |  0.9759
| acc         |  ce.acc  | acc.n2   |       1       |   20   |  0.8956
| acc         |  ce.acc  | acc.n3   |       1       |   20   |  0.4983
| acc         |  ce.acc  | acc.n4   |       1       |   20   |  0.9535

| don         |  ce.don  | don.n1   |       1       |   20   |  0.9467
| don         |  ce.don  | don.n2   |       1       |   20   |  0.8790
| don         |  ce.don  | don.n3   |       1       |    5   |  0.9524
| acc         |  acc.hi  | acc.fake |       0       |   20   |  0.9808

## Differentiating between organisms


## C. elegans

| splice site | positive | negative | hidden layers | epochs | testing accuracy |
|:-----------:|:--------:|:--------:|:-------------:|:------:|:--------:|
| acc         |  acc.hi  | acc.fake |       1       |   20   |  0.9803
| don         |  don.hi  | don.fake |       1       |   20   |  0.9772
| acc         |  acc.lo  | acc.fake |       1       |    5   |  0.9532
| don         |  don.lo  | don.fake |       1       |    5   |  0.9524
| acc         |  acc.hi  | acc.fake |       0       |   20   |  0.9808


## Fabricated
| splice site | positive | negative | hidden layers | epochs | testing accuracy |
|:-----------:|:--------:|:--------:|:-------------:|:------:|:--------:|
| acc         | acc.ex3  | acc.not  |      0        |  20    |  0.9824
| acc         | acc.ex3  | acc.not  |      1        |  20    |  0.9815
