import datetime
import gc
import numpy
import time
import torch
import torch.utils
import torch.utils.data

N = 16777216
NBITS = 32

def Compress(x, n):
	output = x.reshape([-1, 1])
	indices = torch.sort(torch.abs(output - 0.5), dim=0, descending=True)[1].reshape(-1)
	return (indices.unsqueeze(1)[:n].int(), output[indices][:n].half())

def Decompress(x, size):
	output = torch.full((size * 2 * 126 * 128, ), 0.5, dtype=torch.float16).cuda()
	output[x[0].long()] = x[1].half()
	return output.reshape([size, 2, 126, 128])

class Encoder(torch.nn.Module):
	def __init__(self, feedback_bits, data_size=10000):
		super(Encoder, self).__init__()
		self.indices = torch.nn.Parameter(torch.zeros((N, 1), dtype=torch.int), requires_grad=False)
		self.data = torch.nn.Parameter(torch.zeros((N, 1), dtype=torch.half), requires_grad=False)
		self.decompressed_data = None
		self.feedback_bits = feedback_bits
		self.data_size = data_size
		self.flag = 0
	
	def build(self, data):
		self.data_size = len(data)
		indices, data = Compress(data, N)
		self.indices = torch.nn.Parameter(indices, requires_grad=False)
		self.data = torch.nn.Parameter(data, requires_grad=False)
		
	def initialize(self, device):
		decompressed_data = Decompress((self.indices, self.data.data), int(self.data_size))
		del self.indices
		del self.data
		self.data = decompressed_data.reshape([-1, 2, 126, 16, 8]).permute(0, 3, 1, 2, 4).reshape(-1, 2, 126, 8)
		del decompressed_data
		self.data = torch.cat([torch.full([1, 2, 126, 8], 0.5, dtype=torch.float16, device=device), self.data])
		self.data = torch.cat([self.data, 1 - self.data]).unsqueeze(3)
		
		self.mse_output = torch.zeros([len(self.data), 2, 126, 16, 8], dtype=torch.float16, device=device)
		self.sum_output = torch.zeros([len(self.data), 16], dtype=torch.float16, device=device)
		self.all_sum_output = torch.zeros([2 * len(self.data), 16], dtype=torch.float16, device=device)
		self.integer_output = torch.zeros([64, 16], dtype=torch.int, device=device)
		self.output = torch.zeros([64, 16, 32], dtype=torch.int, device=device)
		
		self.flag = 1
		
	def forward(self, x):
		if self.flag == 0:
			self.initialize(x.device)
		
		x = x.half()
		for a in range(len(x)):
			torch.subtract(x[a].reshape([-1, 2, 126, 16, 8]), self.data, out=self.mse_output)
			torch.pow(self.mse_output, 2, out=self.mse_output)
			torch.sum(self.mse_output, dim=(1, 2, 4), out=self.sum_output)
			self.all_sum_output[:len(self.data)] = self.sum_output
			
			torch.subtract(x[a].reshape([-1, 2, 126, 16, 8]), 1 - self.data, out=self.mse_output)
			torch.pow(self.mse_output, 2, out=self.mse_output)
			torch.sum(self.mse_output, dim=(1, 2, 4), out=self.sum_output)
			self.all_sum_output[len(self.data):] = self.sum_output
			
			self.integer_output[a] = torch.argmin(self.all_sum_output, dim=0)
			
		for a in range(NBITS):
			self.output[:, :, a] = self.integer_output % 2
			torch.bitwise_right_shift(self.integer_output, 1, out=self.integer_output)

		return self.output.reshape([-1, 512])[:len(x)]

class Decoder(torch.nn.Module):
	def __init__(self, feedback_bits, data_size=10000):
		super(Decoder, self).__init__()
		self.indices = torch.nn.Parameter(torch.zeros((N, 1), dtype=torch.int), requires_grad=False)
		self.data = torch.nn.Parameter(torch.zeros((N, 1), dtype=torch.half), requires_grad=False)
		self.feedback_bits = feedback_bits
		self.data_size = data_size
		self.flag = 0
	
	def build(self, data):
		self.data_size = len(data)
		indices, data = Compress(data, N)
		self.indices = torch.nn.Parameter(indices, requires_grad=False)
		self.data = torch.nn.Parameter(data, requires_grad=False)
		
	def initialize(self, device):
		decompressed_data = Decompress((self.indices, self.data.data), int(self.data_size))
		del self.indices
		del self.data
		self.data = decompressed_data.reshape([-1, 2, 126, 16, 8])
		del decompressed_data
		self.data = torch.cat([torch.full([1, 2, 126, 8], 0.5, dtype=torch.float16, device=device), self.data])
		self.data = torch.cat([self.data, 1 - self.data])

		self.flag = 1
	
	def forward(self, x):
		if self.flag == 0:
			self.initialize(x.device)
			
		x = x.reshape([-1, 512 // NBITS, NBITS]).long()
		bit = 1
		bit_output = torch.zeros([len(x), 16], dtype=torch.long, device=x.device)
		for a in range(NBITS):
			bit_output += bit * x[:, :, a]
			bit <<= 1
		
		return (0.5 + ((-1) ** (bit_output > len(self.data)).int()).reshape(-1, 16, 1, 1, 1) * (self.data[bit_output % len(self.data)] - 0.5)).permute(0, 2, 3, 1, 4).reshape(-1, 2, 126, 128)
		
	
class AutoEncoder(torch.nn.Module):
	def __init__(self, feedback_bits):
		super(AutoEncoder, self).__init__()
		self.encoder = Encoder(feedback_bits)
		self.decoder = Decoder(feedback_bits)
	
	def forward(self, x):
		feature = self.encoder(x)
		out = self.decoder(feature)
		return out
  
