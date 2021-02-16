
networks = []

# single perceptron
networks.append(('168', '1'))

# 2 layer perceptron
h = ['1', '2', '4', '8', '16', '32', '64']
for n in h:
	networks.append(('168', n, '1'))

# 3 layer perceptron
for h1 in h:
	for h2 in h:
		networks.append(('168', h1, h2, '1'))

# main loop - unfinished
for net in networks:
	files = f'--file1 ___ --file0 ___'
	layers = f'--layers {" ".join(net)}'
	print(f'python3 mlp.py {files} {layers}')

# other hyperparameters
# learning rate
# momentum