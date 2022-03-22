import datetime
import gc
import numpy
import time
import torch
import torch.utils
import torch.utils.data

N = 16777216
NBITS = 22
ENCODING_BATCH_SIZE = 32768
SELECTED_ROWS = list(range(19, 67))

def Compress(x, n):
	output = x.reshape([-1, 1])
	indices = torch.sort(torch.abs(output - 0.5), dim=0, descending=True)[1].reshape(-1)
	return (indices.unsqueeze(1)[:n].int(), output[indices][:n].half())

def Decompress(x, size):
	output = torch.full((size * 2 * len(SELECTED_ROWS) * 128, ), 0.5, dtype=torch.float16).cuda()
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
		self.flag = 0
	
	def build(self, data):
		self.data_size = len(data)
		indices, data = Compress(data[:, :, SELECTED_ROWS], N)
		self.indices = torch.nn.Parameter(indices, requires_grad=False)
		self.data = torch.nn.Parameter(data, requires_grad=False)
		
	def initialize(self, device):
		decompressed_data = Decompress((self.indices, self.data.data), int(self.data_size))
		del self.indices
		del self.data
		self.data = decompressed_data.reshape([-1, 2, len(SELECTED_ROWS), 16, 8]).permute(0, 3, 1, 2, 4).reshape(-1, 2, len(SELECTED_ROWS), 8)
		del decompressed_data
		self.data = torch.cat(
			[self.data]
			 + [(0.5 * self.data.reshape([-1, 16, 2, len(SELECTED_ROWS), 8])[:, [a]] + 0.5 * self.data.reshape([-1, 16, 2, len(SELECTED_ROWS), 8])[:, [b]]).reshape([-1, 2, len(SELECTED_ROWS), 8]) for a in range(15) for b in range(1 + a, min(16, 4 + a))]
			 + [(0.5 + 0.5 * self.data.reshape([-1, 16, 2, len(SELECTED_ROWS), 8])[:, [a]] - 0.5 * self.data.reshape([-1, 16, 2, len(SELECTED_ROWS), 8])[:, [b]]).reshape([-1, 2, len(SELECTED_ROWS), 8]) for a in range(15) for b in range(1 + a, min(16, 4 + a))]
		, dim=0)
		self.data = torch.cat([
			torch.full([1, 2, len(SELECTED_ROWS), 8], 0.5, dtype=torch.float16, device=device)
			, self.data
			, torch.zeros([(1 + len(self.data) // ENCODING_BATCH_SIZE) * ENCODING_BATCH_SIZE - len(self.data) - 1, 2, len(SELECTED_ROWS), 8], dtype=torch.float16, device=device)
		])[:(2 ** (NBITS - 1))].unsqueeze(3)
		
		self.mse_output = torch.zeros([ENCODING_BATCH_SIZE, 2, len(SELECTED_ROWS), 16, 8], dtype=torch.float16, device=device)
		self.sum_output = torch.zeros([ENCODING_BATCH_SIZE, 16, 8], dtype=torch.float16, device=device)
		self.all_sum_output_1 = torch.zeros([2 * len(self.data), 7], dtype=torch.float16, device=device)
		self.all_sum_output_2 = torch.zeros([2 * len(self.data), 7], dtype=torch.float16, device=device)
		self.all_sum_output_3 = torch.zeros([2 * len(self.data), 9], dtype=torch.float16, device=device)
		self.integer_output_1 = torch.zeros([64, 7], dtype=torch.int, device=device)
		self.integer_output_2 = torch.zeros([64, 7], dtype=torch.int, device=device)
		self.integer_output_3 = torch.zeros([64, 9], dtype=torch.int, device=device)
		self.output_1 = torch.zeros([64, 7, NBITS], dtype=torch.int, device=device)
		self.output_2 = torch.zeros([64, 7, NBITS], dtype=torch.int, device=device)
		self.output_3 = torch.zeros([64, 9, NBITS], dtype=torch.int, device=device)
		
		self.flag = 1
	
	def forward(self, x):
		if self.flag == 0:
			self.initialize(x.device)
			
		x = x[:, :, SELECTED_ROWS].half().reshape([-1, 2, len(SELECTED_ROWS), 16, 8])
		for a in range(len(x)):
			for b in range(0, len(self.data), ENCODING_BATCH_SIZE):
				torch.subtract(x[a], self.data[b:(b + ENCODING_BATCH_SIZE)], out=self.mse_output)
				torch.pow(self.mse_output, 2, out=self.mse_output)
				torch.sum(self.mse_output, dim=(1, 2), out=self.sum_output)
				self.all_sum_output_1[b:(b + ENCODING_BATCH_SIZE)] = torch.sum(self.sum_output[:, :7, :4], dim=2)
				self.all_sum_output_2[b:(b + ENCODING_BATCH_SIZE)] = torch.sum(self.sum_output[:, :7, 4:], dim=2)
				self.all_sum_output_3[b:(b + ENCODING_BATCH_SIZE)] = torch.sum(self.sum_output[:, 7:], dim=2)
				
				torch.subtract(x[a], 1 - self.data[b:(b + ENCODING_BATCH_SIZE)], out=self.mse_output)
				torch.pow(self.mse_output, 2, out=self.mse_output)
				torch.sum(self.mse_output, dim=(1, 2), out=self.sum_output)
				self.all_sum_output_1[(len(self.data) + b):(len(self.data) + b + ENCODING_BATCH_SIZE)] = torch.sum(self.sum_output[:, :7, :4], dim=2)
				self.all_sum_output_2[(len(self.data) + b):(len(self.data) + b + ENCODING_BATCH_SIZE)] = torch.sum(self.sum_output[:, :7, 4:], dim=2)
				self.all_sum_output_3[(len(self.data) + b):(len(self.data) + b + ENCODING_BATCH_SIZE)] = torch.sum(self.sum_output[:, 7:], dim=2)
		
			self.integer_output_1[a] = torch.argmin(self.all_sum_output_1, dim=0)
			self.integer_output_2[a] = torch.argmin(self.all_sum_output_2, dim=0)
			self.integer_output_3[a] = torch.argmin(self.all_sum_output_3, dim=0)
			
		for a in range(NBITS):
			self.output_1[:, :, a] = self.integer_output_1 % 2
			torch.floor_divide(self.integer_output_1, 2, out=self.integer_output_1)
			self.output_2[:, :, a] = self.integer_output_2 % 2
			torch.floor_divide(self.integer_output_2, 2, out=self.integer_output_2)
			self.output_3[:, :, a] = self.integer_output_3 % 2
			torch.floor_divide(self.integer_output_3, 2, out=self.integer_output_3)

		return torch.cat([
			torch.zeros([64, 6], dtype=torch.float, device=x.device)
			, torch.cat([self.output_1, self.output_2, self.output_3], dim=1).reshape([-1, 506]).float()
		], dim=1)[:len(x)]
	
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
		indices, data = Compress(data[:, :, SELECTED_ROWS], N)
		self.indices = torch.nn.Parameter(indices, requires_grad=False)
		self.data = torch.nn.Parameter(data, requires_grad=False)
	
	def initialize(self, device):
		decompressed_data = Decompress((self.indices, self.data.data), int(self.data_size))
		del self.indices
		del self.data
		self.data = decompressed_data.reshape([-1, 2, len(SELECTED_ROWS), 16, 8]).permute(0, 3, 1, 2, 4).reshape(-1, 2, len(SELECTED_ROWS), 8)
		del decompressed_data
		self.data = torch.cat(
			[self.data]
			 + [(0.5 * self.data.reshape([-1, 16, 2, len(SELECTED_ROWS), 8])[:, [a]] + 0.5 * self.data.reshape([-1, 16, 2, len(SELECTED_ROWS), 8])[:, [b]]).reshape([-1, 2, len(SELECTED_ROWS), 8]) for a in range(15) for b in range(1 + a, min(16, 4 + a))]
			 + [(0.5 + 0.5 * self.data.reshape([-1, 16, 2, len(SELECTED_ROWS), 8])[:, [a]] - 0.5 * self.data.reshape([-1, 16, 2, len(SELECTED_ROWS), 8])[:, [b]]).reshape([-1, 2, len(SELECTED_ROWS), 8]) for a in range(15) for b in range(1 + a, min(16, 4 + a))]
		, dim=0)
		self.data = torch.cat([
			torch.full([1, 2, len(SELECTED_ROWS), 8], 0.5, dtype=torch.float16, device=device)
			, self.data
			, torch.zeros([(1 + len(self.data) // ENCODING_BATCH_SIZE) *  ENCODING_BATCH_SIZE - len(self.data) - 1, 2, len(SELECTED_ROWS), 8], dtype=torch.float16, device=device)
		])[:(2 ** (NBITS - 1))]
		
		self.flag = 1

	def forward(self, x):
		if self.flag == 0:
			self.initialize(x.device)

		x = x[:, 6:].reshape([-1, 23, NBITS]).long()
		bit = 1
		bit_output_1 = torch.zeros([len(x), 7], dtype=torch.long, device=x.device)
		bit_output_2 = torch.zeros([len(x), 7], dtype=torch.long, device=x.device)
		bit_output_3 = torch.zeros([len(x), 9], dtype=torch.long, device=x.device)
		for a in range(NBITS):
			bit_output_1 += bit * x[:, :7, a]
			bit_output_2 += bit * x[:, 7:14, a]
			bit_output_3 += bit * x[:, 14:, a]
			bit <<= 1
			
		output_1 = torch.full([len(x), 2, 126, 7, 4], 0.5, dtype=torch.float16, device=x.device)
		output_1[:, :, SELECTED_ROWS] = (0.5 + ((-1) ** (bit_output_1 > len(self.data)).int()).reshape(-1, 7, 1, 1, 1) * (self.data[bit_output_1 % len(self.data), :, :, :4] - 0.5)).permute(0, 2, 3, 1, 4)
		output_2 = torch.full([len(x), 2, 126, 7, 4], 0.5, dtype=torch.float16, device=x.device)
		output_2[:, :, SELECTED_ROWS] = (0.5 + ((-1) ** (bit_output_2 > len(self.data)).int()).reshape(-1, 7, 1, 1, 1) * (self.data[bit_output_2 % len(self.data), :, :, 4:] - 0.5)).permute(0, 2, 3, 1, 4)
		output_3 = torch.full([len(x), 2, 126, 9, 8], 0.5, dtype=torch.float16, device=x.device)
		output_3[:, :, SELECTED_ROWS] = (0.5 + ((-1) ** (bit_output_3 > len(self.data)).int()).reshape(-1, 9, 1, 1, 1) * (self.data[bit_output_3 % len(self.data)] - 0.5)).permute(0, 2, 3, 1, 4)

		return torch.cat([torch.cat([output_1, output_2], dim=4), output_3], dim=3).reshape([-1, 2, 126, 128]).float()

class AutoEncoder(torch.nn.Module):
	def __init__(self, feedback_bits):
		super(AutoEncoder, self).__init__()
		self.encoder = Encoder(feedback_bits)
		self.decoder = Decoder(feedback_bits)
	
	def forward(self, x):
		feature = self.encoder(x)
		out = self.decoder(feature)
		return out
  
