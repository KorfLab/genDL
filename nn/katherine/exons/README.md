README for 4 layer MLP on exon data
==================================

## Performance ##

Topology: 153 -> 15 -> 10 -> 1

| File1 | File0 |   MLP  |   PWM  | Notes
|:-----:|:------|:-------|:-------|:--------------
| cdsi  | cdsf  | 0.9804 |       |
| cds0  | cdsf  | 0.6782 |       | severe overfitting at 100 epochs
| cds+  | cdsf  | 0.8016 |       |
| a.obs |       | 0.8899 |       |
| a.ex3 |       | 0.9918 |       |
| a.ex4 |       | 0.8547 |       |
| cds+  | cds-  |  |        |
| a.ex6 |       | 0.8234 | 0.8483 |
