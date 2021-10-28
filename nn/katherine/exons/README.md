README for MLP on exon data
==================================

## Performance ##

Topology: 153 -> 15 -> 10 -> 1

Table 1: Test Run
| File1 | File0 | Train  |  Test  | Notes
|:-----:|:------|:-------|:-------|:--------------
| cdsi  | cdsf  | 0.9804 |        |
| cds0  | cdsf  | 0.6782 |        | severe overfitting at 100 epochs
| cds+  | cdsf  | 0.5980 |        | severe overfitting at 100 epochs
| cds+  | cds-  |        |        |

Table 2: Find # of epochs where accuracy converges and find a good learning rate
| File1 | File0 | Train  |  Test  | Notes
|:-----:|:------|:-------|:-------|:--------------
| cdsi  | cdsf  |        |        |
| cds0  | cdsf  |        |        |
| cds+  | cdsf  |        |        |
| cds+  | cds-  |        |        |

Table 3: With dropout
| File1 | File0 | Train  |  Test  | Notes
|:-----:|:------|:-------|:-------|:--------------
| cdsi  | cdsf  |        |        |
| cds0  | cdsf  |        |        |
| cds+  | cdsf  |        |        |
| cds+  | cds-  |        |        |

Table 4: With regularization (L1 or L2?)
| File1 | File0 | Train  |  Test  | Notes
|:-----:|:------|:-------|:-------|:--------------
| cdsi  | cdsf  |        |        |
| cds0  | cdsf  |        |        |
| cds+  | cdsf  |        |        |
| cds+  | cds-  |        |        |

Table 5: With dropout and regularization
| File1 | File0 | Train  |  Test  | Notes
|:-----:|:------|:-------|:-------|:--------------
| cdsi  | cdsf  |        |        |
| cds0  | cdsf  |        |        |
| cds+  | cdsf  |        |        |
| cds+  | cds-  |        |        |
