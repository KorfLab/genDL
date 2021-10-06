README for Exon Markov models
=============================

## PWM ##

`strawpwm.py` is pretty much exactly `strawman.py` from the pwm. As you can see
from the table below, the performance isn't very good in a real scenario.


| file1 | file0 | model | perf  | Notes
|:-----:|:-----:|:-----:|:-----:|:------
| cdsi  | cds-  | pwm   | 0.985 | ATG clearly detected
| cds0  | cds-  | pwm   | 0.692 |
| cds+  | cds-  | pwm   | 0.604 |
| cdsi  | cdsf  | pwm   | 0.986 |
| cds0  | cdsf  | pwm   | 0.687 |
| cds+  | cdsf  | pwm   | 0.500 |

## Markov model ##

`strawmm.py` is a quick-n-dirty k-mer score. Once the k-mer size is 3 or
higher, the accuracy gets better. The cdsf dataset (fake sequences) is easier
to discriminate against than cds- (exons on negative strand).

| file1 | file0 | model | perf  | Notes
|:-----:|:-----:|:-----:|:-----:|:------
| cdsi  | cds-  | mm:1  | 0.518 |
| cdsi  | cds-  | mm:2  | 0.595 |
| cdsi  | cds-  | mm:3  | 0.757 |
| cdsi  | cds-  | mm:4  | 0.768 |
| cdsi  | cds-  | mm:5  | 0.773 |
| cdsi  | cdsf  | mm:1  | 0.636 |
| cdsi  | cdsf  | mm:2  | 0.777 |
| cdsi  | cdsf  | mm:3  | 0.820 |
| cdsi  | cdsf  | mm:4  | 0.837 |
| cdsi  | cdsf  | mm:5  | 0.847 |
| cds0  | cds-  | mm:4  | 0.707 |
| cds0  | cdsf  | mm:4  | 0.810 |
| cds+  | cds-  | mm:4  | 0.705 |
| cds+  | cdsf  | mm:4  | 0.809 |

