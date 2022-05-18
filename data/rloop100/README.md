README for ime50 data
=====================

Results
-------

| file1 | file0 |  K1  |  K2  |  K3  |  K4  |  K5  | mlp1 | mlp2 | mlp3 |
|:-----:|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| pos   | neg1  | 81.7 | 90.4 | 93.1 | 95.4 | 96.0 | 81.1 | 91.0 | 91.0 |
| pos   | neg2  | 74.2 | 89.3 | 91.6 | 93.0 | 93.6 | 71.2 | 86.2 | 86.5 |
| pos   | neg3  | 83.0 | 83.0 | 83.1 | 83.1 | 82.9 | 82.1 | 76.7 | 78.2 |
| pos   | neg4  | 49.6 | 87.2 | 89.2 | 90.5 | 90.9 | 50.3 | 76.4 | 76.9 |

+ mlp1.py --layers 400 1
+ mlp2.py --layers 400 100 1
+ mlp3.py --layers 400 200 100 1

+ neg1 - unbiased random
+ neg2 - GC-biased random
+ neg3 - negative strand
+ neg4 - shuffled
