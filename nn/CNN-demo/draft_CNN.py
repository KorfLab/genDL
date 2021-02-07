import argparse
import sys

from gendl import pwm, seqio

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='CNN model')
	parser.add_argument('--path1', required=True, type=str,
		metavar='<path>', help='path to fasta file1')
	parser.add_argument('--path0', required=True, type=str,
		metavar='<path>', help='path to fasta file0')
	parser.add_argument('--split', required=False, type=float, default = 0.2,
		metavar = '<float>', help = 'split size')
	parser.add_argument('--epoch', required=False, type=int, default = 10,
		metavar='<int>', help='number of epochs')
	parser.add_argument('--batch', required=False, type=int, default = 1,
		metavar='<int>', help='batch size')
	parser.add_argument('--seed', required=False, type=int,
		metavar='<int>', help='random seed')
	arg = parser.parse_args()

	assert(0 < arg.split <1 .0)
	assert(type(arg.epoch) == int)
	assert(type(arg.batch) == int)

	if arg.seed:
		random.seed(arg.seed)

	seqs1 = [(1, seq) for name, seq in seqio.read_fasta(arg.file1)]
	seqs0 = [(0, seq) for name, seq in seqio.read_fasta(arg.file0)]
	seqs = seqs1 + seqs0
	random.shuffle(seqs)

	#specify batch_size


	class CNN(nn.Module):
		def __init__(self):
			super(CNN, self).__init__()
			self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
			self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
			self.conv3 = nn.Conv2d(32,64, kernel_size=5)
			self.fc1 = nn.Linear(3*3*64, 256)
			self.fc2 = nn.Linear(256, 10)

		def forward(self, x):
			x = F.relu(self.conv1(x))
			#x = F.dropout(x, p=0.5, training=self.training)
			x = F.relu(F.max_pool2d(self.conv2(x), 2))
			x = F.dropout(x, p=0.5, training=self.training)
			x = F.relu(F.max_pool2d(self.conv3(x),2))
			x = F.dropout(x, p=0.5, training=self.training)
			x = x.view(-1,3*3*64 )
			x = F.relu(self.fc1(x))
			x = F.dropout(x, training=self.training)
			x = self.fc2(x)
			return F.log_softmax(x, dim=1)
