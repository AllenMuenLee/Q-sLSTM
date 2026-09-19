import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler


from torch.utils.data import Dataset, DataLoader


# create a figure window
# fig = plt.figure(1, figsize=(9,8))


x = np.arange(-2, 20, 0.01)
data = 0.
for n in range(11):
	data += np.exp(-10*(x-2*n)**2)*np.exp(-x/16)
# plt.plot(x, data)
# plt.show()

# print(data)
# print(len(data))


def generate_dataset(data_src = data):
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_dataset = scaler.fit_transform(data_src.reshape(-1, 1))
	return scaled_dataset 

def transform_data_single_predict(data, seq_length):
	x = []
	y = []

	for i in range(len(data)-seq_length-1):
		_x = data[i:(i+seq_length)]
		_y = data[i+seq_length]
		x.append(_x)
		y.append(_y)
	x_var = Variable(torch.from_numpy(np.array(x).reshape(-1, seq_length)).float())
	y_var = Variable(torch.from_numpy(np.array(y)).float())

	return x_var, y_var


def get_delayed_quantum_control_data(data = data, seq_len = 4):
	scaled_dataset = generate_dataset(data)
	return transform_data_single_predict(data = scaled_dataset, seq_length = seq_len)



###
class DelayedQCDataset(Dataset):
	"""
	將單一 time-series（你的 data）切成 (seq_len -> 下一點) 的 PyTorch Dataset。
	- 預設 pre_scaled=False：會在內部用 MinMaxScaler 將資料縮放到 [-1, 1]
	- 若你已經先縮放好資料，設 pre_scaled=True
	- 輸出:
		x: [N, seq_len]
		y: [N]
	"""
	def __init__(self,
				 data_src: np.ndarray = data,
				 seq_len: int = 4,
				 pre_scaled: bool = False,
				 feature_range = (-1, 1),
				 dtype = torch.float32):
		self.seq_len = int(seq_len)
		self.dtype = dtype

		arr = np.asarray(data_src).reshape(-1)

		# 縮放控制
		self.scaler = None
		if pre_scaled:
			scaled = arr
		else:
			self.scaler = MinMaxScaler(feature_range=feature_range)
			scaled = self.scaler.fit_transform(arr.reshape(-1, 1)).reshape(-1)

		# 依你的 transform 規則切 (x, y)
		xs, ys = [], []
		for i in range(len(scaled) - self.seq_len - 1):
			xs.append(scaled[i : i + self.seq_len])
			ys.append(scaled[i + self.seq_len])

		self.x = torch.tensor(np.array(xs), dtype=self.dtype)   # [N, seq_len]
		self.y = torch.tensor(np.array(ys), dtype=self.dtype)   # [N]

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, idx):
		return self.x[idx].unsqueeze(-1), self.y[idx] # unsqueeze(-1) for LSTM-like models

	def inverse_transform_y(self, y_tensor: torch.Tensor):
		"""
		若本 Dataset 有做縮放（pre_scaled=False），把 y 從縮放域轉回原尺度；
		若 pre_scaled=True，直接回傳原值。
		"""
		if self.scaler is None:
			return y_tensor
		y_np = y_tensor.detach().cpu().numpy().reshape(-1, 1)
		inv = self.scaler.inverse_transform(y_np).reshape(-1)
		return torch.tensor(inv, dtype=y_tensor.dtype)


def make_delayed_qc_dataset(data_src=data,
							seq_len: int = 4,
							batch_size: int = 64,
							shuffle: bool = True,
							pre_scaled: bool = False,
							feature_range = (-1, 1),
							dtype = torch.float32):
	"""
	工廠函式：把 Delayed Quantum Control 的序列包成 Dataset + DataLoader。
	預設會在內部縮放（pre_scaled=False），如同你原本的 get_* 做法。
	"""
	ds = DelayedQCDataset(
		data_src=data_src,
		seq_len=seq_len,
		pre_scaled=pre_scaled,
		feature_range=feature_range,
		dtype=dtype
	)
	dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
	return ds, dl
	

###
def plotting_test(data_src):
	ax1 = fig.add_subplot()
	ax1.plot(x,data_src)
	ax1.axhline(color="grey", ls="--", zorder=-1)
	ax1.set_ylim(-1,1)
	ax1.text(0.5, 0.95,'Delayed Quantum Control', ha='center', va='top',
		 transform = ax1.transAxes)

	plt.show()

def main():
	scaled_dataset = generate_dataset(data)
	plotting_test(scaled_dataset)
	x, y = get_delayed_quantum_control_data(data)
	print(x)
	print(y)

	

if __name__ == '__main__':
	main()