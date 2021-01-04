#!/usr/bin/env python3

import sys
import os
import re
import gzip

import data_shaper as ds
import numpy as np

path = os.path.abspath(os.path.dirname(__file__))
figs_path = os.path.join(path, '../figs')
assert(os.path.isdir(figs_path))

"""
acc = dict()
acc['hi'] = dict()
acc['lo'] = dict()
acc['fake'] = dict()
don = dict()
don['hi'] = dict()
don['lo'] = dict()
don['fake'] = dict()
"""

summary = """
| File                | Sequences |
|:--------------------|:----------|
"""

for filename in sorted(os.listdir(os.getcwd())):
	if re.search(r'hilo', filename):
		continue
	
	if re.search(r'gz$', filename):
		f = filename.split('.')
		seqs = ds.read_data(filename, 42)
		x = str(len(list(seqs)))
		summary += f"""| {filename:19s} | {x:9s} |\n"""
		

print(summary)
"""
| File            | Sequences |
|:----------------|:----------|
| acc.fake.txt    |  210674   |
| acc.lo.true.txt |   10432   |
| acc.hi.true.txt |    9196   |
| don.fake.txt    |  194418   |
| don.lo.true.txt |    9862   |
| don.hi.true.txt |    9220   |
"""