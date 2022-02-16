#!/usr/bin/python3

"""
Shared module for PyTorch NN training
"""

from math import floor
import random
import sys
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import datetime
import numpy as np
import torch
import torch.utils.data as data_utils

from gendl import seqio

# Dictionary for default values for convolutions in encoder networks
cnn_template = {
	'channels'      : 1,
	'conv_ks'       : (1, 1),
	'pool_ks'       : (1, 1),
	'conv_paddings' : (0, 0),
	'pool_paddings' : (0, 0),
	'conv_strides'  : (1, 1),
	'pool_strides'  : (1, 1)
}


def make_1d_loaders(
	file1=None,
	file0=None,
	num=None,
	start=None,
	stop=None,
	split=None,
	batch=None,
	seed=None):
	
	s1 = seqio.fasta2onehot(file1, 1, start=start, stop=stop)
	s2 = seqio.fasta2onehot(file0, 0, start=start, stop=stop)
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed=seed)
	
	pos_seqs   = []
	pos_labels = []
	neg_seqs   = []
	neg_labels = []
	for ps in s1:
		#seq = ps[:-1]
		seq = ps
		seq = list(seq)
		seq = np.array(seq, dtype=np.float64)
		pos_seqs.append(seq)
	
	for ns in s2:
		#seq = ns[:-1]
		seq = ns
		seq = list(seq)
		seq = np.array(seq, dtype=np.float64)
		neg_seqs.append(seq)
	
	random.shuffle(pos_seqs)
	random.shuffle(neg_seqs)
	
	pos_seqs = pos_seqs[:num]
	neg_seqs = pos_seqs[:num]
	
	pos_seqs = np.array(pos_seqs)
	neg_seqs = np.array(neg_seqs)
	
	# find how large train/test will be given arg.split
	test_size = floor(num * split)
	
	s1_train = pos_seqs[test_size:]
	s1_test  = pos_seqs[:test_size]
	s2_train = neg_seqs[test_size:]
	s2_test  = neg_seqs[:test_size]
	
	# put all train/test sequences together
	train_sequences = np.concatenate((s1_train, s2_train), axis=0)
	test_sequences  = np.concatenate((s1_test, s2_test),   axis=0)
	
	# put all train/test labels together
	train_labels = np.ones(s1_train.shape[0])
	train_labels = np.concatenate(
		(train_labels, np.zeros(s2_train.shape[0])),
		axis=0
	)
	train_labels = train_labels.reshape(train_labels.shape[0], 1)
	
	test_labels = np.ones(s1_test.shape[0])
	test_labels = np.concatenate(
		(test_labels, np.zeros(s2_test.shape[0])),
		axis=0
	)
	test_labels = test_labels.reshape(test_labels.shape[0], 1)
	
	train = data_utils.TensorDataset(
		torch.Tensor(train_sequences),
		torch.Tensor(train_labels)
	)
	
	train_loader = data_utils.DataLoader(
		train,
		batch_size=batch,
		shuffle=True
	)
	
	test = data_utils.TensorDataset(
		torch.Tensor(test_sequences),
		torch.Tensor(test_labels)
	)
	
	test_loader = data_utils.DataLoader(
		test,
		batch_size=1,
		shuffle=True
	)
	
	return train_loader, test_loader


def make_2d_loaders(
	file1=None,
	file0=None,
	num=None,
	start=None,
	stop=None,
	split=None,
	batch=None,
	seed=None):
	
	s1 = seqio.seq2features(
		seqs=file1,
		num=num,
		start=start,
		stop=stop,
		seed=seed,
		label=1
	)
	s1 = s1.reshape(s1.shape[0], 1, s1.shape[1], s1.shape[2])
	
	s2 = seqio.seq2features(
		seqs=file0,
		num=num,
		start=start,
		stop=stop,
		seed=seed,
		label=0
	)
	s2 = s2.reshape(s2.shape[0], 1, s2.shape[1], s2.shape[2])
	
	# find how large train/test will be given arg.split
	test_size = floor(num * split)
	
	s1_train = s1[test_size:]
	s1_test  = s1[:test_size]
	s2_train = s2[test_size:]
	s2_test  = s2[:test_size]
	
	rows = s1_train.shape[2]
	cols = s1_train.shape[3]
	
	# put all train/test sequences together
	train_sequences = np.concatenate((s1_train, s2_train), axis=0)
	test_sequences  = np.concatenate((s1_test, s2_test),   axis=0)
	
	# put all train/test labels together
	train_labels = np.ones(s1_train.shape[0])
	train_labels = np.concatenate(
		(train_labels, np.zeros(s2_train.shape[0])),
		axis=0
	)
	train_labels = train_labels.reshape(train_labels.shape[0], 1)
	
	test_labels = np.ones(s1_test.shape[0])
	test_labels = np.concatenate(
		(test_labels, np.zeros(s2_test.shape[0])),
		axis=0
	)
	test_labels = test_labels.reshape(test_labels.shape[0], 1)
	
	train = data_utils.TensorDataset(
		torch.Tensor(train_sequences),
		torch.Tensor(train_labels)
	)
	
	train_loader = data_utils.DataLoader(
		train,
		batch_size=batch,
		shuffle=True
	)
	
	test = data_utils.TensorDataset(
		torch.Tensor(test_sequences),
		torch.Tensor(test_labels)
	)
	
	test_loader = data_utils.DataLoader(
		test,
		batch_size=1,
		shuffle=True
	)
	
	return train_loader, test_loader


class Dict2Obj(object):
	"""
	Simple class to turn a dictionary to an object.
	Useful for named attributes in convolution/convolution-transpose size
	calculations.
	
	Inputs
	------
	dic:      dictionary to object-ize.
	template: dictionary for default values.
		template dictionary is different dependening on convolution or
		convolution transpose. Only keys in template not in dic are added as
		object attributes.
	
	Returns
	------
	obj: obj with named attributes from input dictionary keys.
	"""
	def __init__(self, dic, template):
		for key in dic:
			setattr(self, key, dic[key])
		for key in template:
			if hasattr(self, key): continue
			setattr(self, key, template[key])


def evaluate_model(outs, labels):
	outs   = np.array(outs)
	print('net', outs)
	labels = np.array(labels)
	outs = np.around(outs)
	print('tru', labels)
	print('cls', outs)
	print('tru', labels)
	print()
	acc  = accuracy_score(labels, outs)
	f1   = f1_score(labels, outs, average='weighted')
	conf = confusion_matrix(labels, outs)
	
	tn = conf[0][0]
	fn = conf[1][0]
	tp = conf[1][1]
	fp = conf[0][1]
	
	return acc, f1, tp, tn, fn, fp


def fit_model(
	model,
	train=None,
	test=None,
	optimizer=None,
	criterion=None,
	device=None,
	epochs=None,
	file=None):
	"""
	Fit function for all Torch Models
	
	Parameters
	----------
	train:     PyTorch dataloader
	test:      PyTorch dataloader
	device:    CPU/GPU device
	optimizer: PyTorch optimizer
	criterion: PyTorch loss
	epochs:    Number of epochs
	
	Returns
	-------
	Returns fitted model
	"""
	
	assert(model is not None)
	assert(hasattr(model, 'forward'))
	assert(train is not None)
	assert(hasattr(train, 'batch_size'))
	assert(type(epochs) == int and epochs > 0)
	assert(device is not None)
	assert(optimizer is not None)
	assert(hasattr(optimizer, 'zero_grad'))
	assert(criterion is not None)
	assert(hasattr(criterion, 'forward'))
	
	if test is not None: assert(hasattr(test, 'batch_size'))
	
	for epoch in range(epochs):
		loss = 0
		predictions = []
		predictions  = np.array(predictions)
		ground_truth = predictions
		model.train()
		for sequences, labels in train:
			sequences = sequences.to(device)
			#print(sequences)
			#print(labels)
			# reset the gradients back to zero
			# PyTorch accumulates gradients on subsequent backward passes
			optimizer.zero_grad()
			
			# perform forward pass on batch
			outputs = model.forward(sequences)
			#print(outputs)
			# compute training loss
			train_loss = criterion(outputs, labels)
			
			# compute accumulated gradients
			train_loss.backward()
				
			# perform parameter update based on current gradients
			optimizer.step()
			
			# add the mini-batch training loss to epoch loss
			loss += train_loss.item()
			
		# compute the epoch training loss
		loss = loss / len(train)
		
		# evaluate model on full training set
		predictions  = []
		ground_truth = []
		model.eval()
		with torch.no_grad():
			for sequences, labels in train:
				outputs = model(sequences)
				predicted = outputs.data
				predicted = predicted.numpy()
				predicted = predicted.reshape(predicted.shape[0])
				predictions.extend(list(predicted))
				labels = labels.numpy()
				labels = labels.reshape(labels.shape[0])
				ground_truth.extend(list(labels))
		
		tacc, tf1, ttp, ttn, tfn, tfp = evaluate_model(
			predictions,
			ground_truth
		)
		
		vloss = 0
		vpredictions  = []
		vground_truth = [] 
		for bv, lv in test:
			bv = bv.to(device)
			
			outputs = model(bv)
			test_loss = criterion(outputs, lv)
			vloss += test_loss.item()
			
			preds = outputs.data
			preds = preds.numpy()
			preds = preds.reshape(preds.shape[0])
			vpredictions.extend(list(preds))
			lv = lv.numpy()
			lv = lv.reshape(lv.shape[0])
			vground_truth.extend(list(lv))
			
		vloss = vloss / len(test)
		vacc, vf1, vtp, vtn, vfn, vfp = evaluate_model(
			vpredictions,
			vground_truth
		)
		# display the epoch training loss
		progress =  f"epoch: {epoch+1}/{epochs} train loss: {loss:.4f} "
		progress += f"acc: {tacc:.4f} f1: {tf1:.4f} tp: {ttp} tn: {ttn} "
		progress += f"fn: {tfn} fp: {tfp}\n"
		progress += f"test loss:  {vloss:.4f}  acc: {vacc:.4f} f1: {vf1:.4f} "
		progress += f"tp: {vtp} tn: {vtn} fn: {vfn} fp: {vfp}\n"
		file.write(progress)
		print(progress)
	return model
	

def layers_list(dic, template):
	"""
	Unroll dictionary that defines a CNN into a list of
	object containers for parameters in each layer.
	`layers_list` enforces that all values in dictionary are the same size.
	
	Input
	-----
	dic: Dictionary describing network layers.
	template: template dictionary with default values for parameters.
	
	Returns
	-------
	layers: list of objects for parameters per layer
	"""
	for v1 in dic.values():
		for v2 in dic.values(): assert(len(v2) == len(v1))
	
	layers = []
	kk = list(dic.keys())
	size = len(dic[kk[0]])
	for ii in range(size):
		new = {}
		for k, v in dic.items():
			if k not in new: new[k] = None
			new[k] = v[ii]
		newobj = Dict2Obj(new, template)
		layers.append(newobj)
	
	return layers


def conv_pool_out(hin, win, ksize=None, padding=None, stride=None):
	"""
	Compute resulting size of feature maps after convolution/pooling.
	
	Parameters
	----------
	* Positional args
		hin: height of input feature matrix
		win: width of input feature matrix
	* Keyword args
		ksize:   tuple for kernel dimensions
		padding: tuple of padding size for both dimensions
		stride:  tuple for stride step in each dimensions
	
	Returns
	------
	hout: resulting height of feature matrix
	wout: resulting width of feature matrix
	"""
	assert(hin is not None and win is not None)
	assert (ksize is not None and padding is not None and stride is not None)
	assert(type(hin) == int and type(win) == int)
	assert(
		type(ksize) == tuple and
		type(padding) == tuple and
		type(stride) == tuple)
	
	for k, p, s in zip(ksize, padding, stride):
		assert(type(k) == int and type(p) == int and type(s) == int)
	
	hout = hin + 2 * padding[0] - ksize[0]
	hout /= stride[0]
	hout += 1
	
	wout = win + 2 * padding[1] - ksize[1]
	wout /= stride[1]
	wout += 1
	
	if hout.is_integer() and wout.is_integer: return int(hout), int(wout)
	else:                                     return None, None


def cnn_validator(hin, win, layers=None):
	"""
	Validate if proposed CNN layers produce valid dimensions for resulting
	feature maps 
	
	Parameters
	----------
	* Positional args
		hin: rows/height of input feature matrix
		win: columns/width of input feature matrix
	* Keyword args
		layers: list of object containers holding parameters for each layer.
	
	Returns
	-------
	True/False if validation is successful or not
	(h, w) height and width of resulting feature map 
	"""
	assert(layers is not None)
	assert(type(hin) == int and type(win) == int)
	
	print(hin, win)
	h = hin
	w = win
	for l in layers:
		h, w = conv_pool_out(h, w, 
			ksize=l.conv_ks,
			padding=l.conv_paddings,
			stride=l.conv_strides)
		print(h, w)
		if h < 0 or w < 0: return False, None
		h, w = conv_pool_out(h, w,
			ksize=l.pool_ks,
			padding=l.pool_paddings,
			stride=l.pool_strides)
		print(h, w)
		if h < 0 or w < 0: return False, None
	
	return True, (h, w)


def model_validator(inshape=None, model=None):
	"""
	Validate CNN model architecture
	
	Parameters:
	inshape: tuple of input matrix dimensions
	model: dictionary of lists describing parameters for convolution+maxpooling
		layers
	
	
	Returns:
	True if network specified is a valid CNN and autoencoder. 
	False otherwise
	"""
	rows, cols = inshape
	
	model_layers = layers_list(model, cnn_template)
	status, latent = cnn_validator(rows, cols, layers=model_layers)
	
	return status


if __name__ == '__main__':
	# Test validator functions
	model1 = {
		'conv_ks':       [(5,4), (5,4)],
		'pool_ks':       [(3,3), (5,5)],
		'conv_paddings': [(2,2), (2,1)],
		'pool_paddings': [(1,1), (2,2)],
	}
	
	print('model1 validation:', end=' ')
	print(model_validator(inshape=(51,4), model=model1))
	
	model2 = {
		'conv_ks': [(3,2), (3,3)],
		'pool_ks': [(2,2), (2,2)]
	}
	
	print('model2 validation:', end=' ')
	print(model_validator(inshape=(51,4), model=model2))











































