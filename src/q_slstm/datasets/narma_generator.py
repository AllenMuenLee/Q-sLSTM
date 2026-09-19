# 2022 08 01
# Add processing code to process the NARMA data to input and target

# 2022 07 28
# NARMA dataset generator generator from https://www.nature.com/articles/s41598-022-05061-w.pdf
# Author: Samuel Yen-Chi Chen



import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

# Plotting function

# create a figure window
# fig = plt.figure(1, figsize=(9,8))


# Load the original dataset
# with open('input_sequence_NARMA.pickle','rb') as f:
# 	input_sequence = pickle.load(f)
# with open('target_sequence_NARMA.pickle','rb') as f:
# 	target_sequence = pickle.load(f)

def transform_data_single_predict(data_input, data_target, seq_length):
	'''
	data_input: time series
	data_target: time series
	'''
	x = []
	y = []

	assert(len(data_input) == len(data_target))
	data_len = len(data_input)
	for i in range(data_len - seq_length-1):
		_x = data_input[i:(i+seq_length)]
		_y = data_target[i+seq_length]
		x.append(_x)
		y.append(_y)
	x_var = Variable(torch.from_numpy(np.array(x).reshape(-1, seq_length)).float())
	y_var = Variable(torch.from_numpy(np.array(y)).float())

	return x_var, y_var




def plotting_test(data_input, data_target, true_input, true_target):
	ax1 = fig.add_subplot()
	ax1.plot(data_input)
	ax1.plot(data_target)
	ax1.plot(true_input)
	ax1.plot(true_target)

	ax1.axhline(color="grey", ls="--", zorder=-1)
	# ax1.set_ylim(-1,1)
	ax1.text(0.5, 0.95,'NARMA', ha='center', va='top',
		 transform = ax1.transAxes)

	plt.show()


def NARMA2_Generator(initial_y = 0, T = 300):
	u = u_generator(alpha_bar = 2.11, beta_bar = 3.73, gamma_bar = 4.11, T = T)
	y = []
	y.append(initial_y)

	for t in range(T):
		if t-1 < 0:
			y_t_1 = 0.4 * y[t] + 0.4 * y[t] * initial_y + 0.6 * (u[t])**3 + 0.1
			y.append(y_t_1)
		else:
			y_t_1 = 0.4 * y[t] + 0.4 * y[t] * y[t-1] + 0.6 * (u[t])**3 + 0.1
			y.append(y_t_1)

	return u, np.array(y[:-1])

def NARMA_n_Generator(alpha = 0.3, beta = 0.05, gamma = 1.5, delta = 0.1, n_0 = 5, T = 300):
	u = u_generator(alpha_bar = 2.11, beta_bar = 3.73, gamma_bar = 4.11, T = T)
	y = []
	y.append(0.196) # Need a y_0 (Need to find a correct initial values)


	u_initial_values = [] # -1, -2, ... , -(n_0 - 1)
	y_initial_values =[] # -1, -2, ... , -(n_0 - 1)
	
	# Set the initial values to be all zero. Do not know the actual values from the paper
	for i in range(1, n_0):
		u_initial_values.append(0) # default initial values are zero
		y_initial_values.append(0) # default initial values are zero

	for t in range(T):

		# Calculate the sum([y[t - j] for j in range(n_0)])
		y_res_temp = 0
		for j in range(n_0):
			if t - j >= 0:
				y_res_temp += y[t - j]
			else:
				y_res_temp += y_initial_values[j - t - 1] # reserse the order, index from 0 of a list

		# Calculate the u[t - n_0]
		u_res_temp = None
		if t - n_0 + 1 >= 0:
			u_res_temp = u[t - n_0 + 1]
		else:
			u_res_temp = u_initial_values[n_0 - t - 1 - 1] # reserse the order, index from 0 of a list

		y_t_1 = alpha * y[t] + beta * y[t] * y_res_temp + gamma * u_res_temp * u[t] + delta
		y.append(y_t_1)


		# When there is no negative index for u[] and y[]
		# y_t_1 = alpha * y[t] + beta * y[t] * (sum([y[t - j] for j in range(n_0)])) + gamma * u[t - n_0] * u[t] + delta
		# y.append(y_t_1)
	return u, np.array(y[:-1])


def u_generator(alpha_bar = 2.11, beta_bar = 3.73, gamma_bar = 4.11, T = 300):
	'''
	Generate the input sequenct u_t
	'''

	res = []
	for t in range(T):
		u_t = 0.1 * (np.sin((2 * np.pi * alpha_bar * t)/T) * np.sin((2 * np.pi * beta_bar * t)/T) * np.sin((2 * np.pi * gamma_bar * t)/T) + 1)
		res.append(u_t)

	return np.array(res)


def get_narma_data(n_0 = 5, seq_len = 4):
	# scaled_dataset = generate_dataset(data)
	if n_0 == 2:
		input_sequence, target_sequence = NARMA2_Generator(initial_y = 0.196)
	else:
		input_sequence, target_sequence = NARMA_n_Generator(n_0 = n_0)


	return transform_data_single_predict(
		data_input = input_sequence, 
		data_target = target_sequence, 
		seq_length = seq_len)




# 2025 09 19: PyTorch Dataset
class NARMADataset(Dataset):
	"""
	將 (input_sequence, target_sequence) 轉成 seq_len->下一步 的 PyTorch Dataset。
	- 預設假設兩者已是最終尺度 (pre_scaled=True)
	- 若 pre_scaled=False，會各自以 MinMaxScaler 縮放到 [-1, 1]
	- 輸出:
		x: [N, seq_len]
		y: [N]
	"""
	def __init__(self,
				 input_sequence: np.ndarray,
				 target_sequence: np.ndarray,
				 seq_len: int = 4,
				 pre_scaled: bool = True,
				 feature_range = (-1, 1),
				 dtype = torch.float32):
		assert len(input_sequence) == len(target_sequence), "input/target 長度需相同"
		self.seq_len = int(seq_len)
		self.dtype = dtype

		in_arr = np.asarray(input_sequence).reshape(-1)
		tg_arr = np.asarray(target_sequence).reshape(-1)

		# 可選縮放（各自一個 scaler）
		self.scaler_x = None
		self.scaler_y = None
		if pre_scaled:
			in_scaled = in_arr
			tg_scaled = tg_arr
		else:
			self.scaler_x = MinMaxScaler(feature_range=feature_range)
			self.scaler_y = MinMaxScaler(feature_range=feature_range)
			in_scaled = self.scaler_x.fit_transform(in_arr.reshape(-1, 1)).reshape(-1)
			tg_scaled = self.scaler_y.fit_transform(tg_arr.reshape(-1, 1)).reshape(-1)

		# 依你的 transform 規則切 (x, y)
		xs, ys = [], []
		data_len = len(in_scaled)
		for i in range(data_len - self.seq_len - 1):
			xs.append(in_scaled[i : i + self.seq_len])
			ys.append(tg_scaled[i + self.seq_len])

		self.x = torch.tensor(np.array(xs), dtype=self.dtype)   # [N, seq_len]
		self.y = torch.tensor(np.array(ys), dtype=self.dtype)   # [N]

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, idx):
		# return self.x[idx], self.y[idx]
		return self.x[idx].unsqueeze(-1), self.y[idx] # unsqueeze(-1) for LSTM-like models

	# 方便把 y 從縮放域轉回原尺度（只有在 pre_scaled=False 時有用）
	def inverse_transform_y(self, y_tensor: torch.Tensor):
		if self.scaler_y is None:
			return y_tensor
		y_np = y_tensor.detach().cpu().numpy().reshape(-1, 1)
		inv = self.scaler_y.inverse_transform(y_np).reshape(-1)
		return torch.tensor(inv, dtype=y_tensor.dtype)


def make_narma_dataset(n_0: int = 5,
					   T: int = 300,
					   seq_len: int = 4,
					   batch_size: int = 32,
					   shuffle: bool = True,
					   pre_scaled: bool = True,
					   feature_range = (-1, 1),
					   dtype = torch.float32,
					   initial_y_for_narma2: float = 0.196):
	"""
	工廠：直接用你的 NARMA 產生器生資料 -> 建 Dataset -> DataLoader。
	- n_0=2 走 NARMA2_Generator(initial_y=...)
	- 其他 n_0 走 NARMA_n_Generator(n_0=n_0, T=T)
	"""
	if n_0 == 2:
		u, y = NARMA2_Generator(initial_y=initial_y_for_narma2, T=T)
	else:
		u, y = NARMA_n_Generator(n_0=n_0, T=T)

	ds = NARMADataset(
		input_sequence=u,
		target_sequence=y,
		seq_len=seq_len,
		pre_scaled=pre_scaled,
		feature_range=feature_range,
		dtype=dtype
	)
	dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
	return ds, dl


def main():
	
	seq = u_generator()
	u_2, seq_2 = NARMA2_Generator(initial_y = 0.196)
	# plotting_test(data_input = seq, data_target = seq_2, true_input = input_sequence, true_target = target_sequence)

	# print out the initial values
	# print(seq[:10])
	# print(seq_2[:10])
	# print(target_sequence[:10])

	u_3, seq_3 = NARMA_n_Generator(n_0 = 5)
	u_4, seq_4 = NARMA_n_Generator(n_0 = 10)

	# plotting_test(data_input = seq, data_target = seq_3, true_input = input_sequence, true_target = target_sequence)
	# plotting_test(data_input = seq, data_target = seq_4, true_input = input_sequence, true_target = target_sequence)
	print(len(seq))
	print(len(seq_2))
	print(len(seq_3))
	print(len(seq_4))

	ds, dl = make_narma_dataset()

	print(ds.x.shape)
	print(ds.y.shape)
	print(ds.x)
	print(ds.y)
	x_var, y_var = get_narma_data(n_0=5, seq_len=4)
	print(x_var)
	print(y_var)
	print(ds.x == x_var)
	print(ds.y == y_var)

	return

if __name__ == '__main__':
	main()