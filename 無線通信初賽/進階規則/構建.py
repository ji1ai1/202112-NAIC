import datetime
import joblib
import numpy
import scipy.io
import sklearn.cluster
import torch

TrainingData = numpy.transpose(scipy.io.loadmat("train/Htrain.mat")["H_train"].astype('float32'), [0, 3, 1, 2])
TestingData = numpy.transpose(scipy.io.loadmat("train/Htest.mat")["H_test"].astype('float32'), [0, 3, 1, 2])
N = 16777216
NBITS = 22
ENCODING_BATCH_SIZE = 16384
SELECTED_ROWS = list(range(19, 67))

def Compress(x, n):
	output = x.reshape([-1, 1])
	indices = torch.sort(torch.abs(output - 0.5), dim=0, descending=True)[1].reshape(-1)
	return (indices.unsqueeze(1)[:n].int(), output[indices][:n].half())

def Decompress(x, size):
	output = torch.full((size * 2 * len(SELECTED_ROWS) * 128,), 0.5, dtype=torch.float16).cuda()
	output[x[0].long()] = x[1].half()
	return output.reshape([size, 2, len(SELECTED_ROWS), 128])

class Encoder(torch.nn.Module):
	def __init__(self, feedback_bits, data_size=10000):
		super(Encoder, self).__init__()
		self.indices = torch.nn.Parameter(torch.zeros((N, 1), dtype=torch.int), requires_grad=False)
		self.data = torch.nn.Parameter(torch.zeros((N, 1), dtype=torch.half), requires_grad=False)
		self.decompressed_data = None
		self.feedback_bits = feedback_bits
		self.data_size = data_size
	
	def build(self, data):
		self.data_size = len(data)
		indices, data = Compress(data[:, :, SELECTED_ROWS], N)
		self.indices = torch.nn.Parameter(indices, requires_grad=False)
		self.data = torch.nn.Parameter(data, requires_grad=False)
	
class Decoder(torch.nn.Module):
	def __init__(self, feedback_bits, data_size=10000):
		super(Decoder, self).__init__()
		self.indices = torch.nn.Parameter(torch.zeros((N, 1), dtype=torch.int), requires_grad=False)
		self.data = torch.nn.Parameter(torch.zeros((N, 1), dtype=torch.half), requires_grad=False)
		self.feedback_bits = feedback_bits
		self.data_size = data_size
	
	def build(self, data):
		self.data_size = len(data)
		indices, data = Compress(data[:, :, SELECTED_ROWS], N)
		self.indices = torch.nn.Parameter(indices, requires_grad=False)
		self.data = torch.nn.Parameter(data, requires_grad=False)

EncoderModel = Encoder(512).cuda()
EncoderModel.build(torch.Tensor(numpy.concatenate([TestingData, TrainingData])).cuda())
torch.save({"state_dict": EncoderModel.state_dict()}, "project/encoder.pth.tar")
del EncoderModel
DecoderModel = Decoder(512).cuda()
DecoderModel.build(torch.Tensor(numpy.concatenate([TestingData, TrainingData])).cuda())
torch.save({"state_dict": DecoderModel.state_dict()}, "project/decoder.pth.tar")
