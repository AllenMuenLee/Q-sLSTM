import numpy as np
import scipy.special
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

# create a figure window
fig = plt.figure(1, figsize=(9,8))

# create arrays for a few Bessel functions and plot them
x = np.linspace(2, 100, 256)
j0 = scipy.special.jn(0, x)
j1 = scipy.special.jn(1, x)
j2 = scipy.special.jn(2, x)
y0 = scipy.special.yn(0, x)
y1 = scipy.special.yn(1, x)
y2 = scipy.special.yn(2, x)



def generate_dataset(data_src = j0):
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

def get_bessel_data(data = j2, seq_len = 4):
	scaled_dataset = generate_dataset(data)
	return transform_data_single_predict(data = scaled_dataset, seq_length = seq_len)

# 2025 09 19: PyTorch DataSet and DataLoader
class BesselSequenceDataset(Dataset):
	"""
	將單序列 (如 j0/j1/...) 轉成 (seq_len -> 下一步) 的 PyTorch Dataset。
	- 輸入: data_src (1D numpy array), 會自動做 MinMaxScaler 到 [-1, 1]
	- 取樣: 連續 seq_len 點當作 x，下一點當作 y
	- 回傳: x shape = (seq_len,)；y shape = ()  (維持你原本的設計)
	"""
	def __init__(self, data_src = j2, seq_len=4, feature_range=(-1, 1), dtype=torch.float32):
		# 存原始資料與 scaler（方便之後反轉）
		self.scaler = MinMaxScaler(feature_range=feature_range)
		# 先確保是 1D np.ndarray
		data_src = np.asarray(data_src).reshape(-1)
		# 做縮放成 2D 再壓回 1D
		scaled = self.scaler.fit_transform(data_src.reshape(-1, 1)).reshape(-1)

		self.seq_len = int(seq_len)
		self.dtype = dtype

		# 預先切好所有 (x, y)
		xs, ys = [], []
		for i in range(len(scaled) - self.seq_len - 1):
			xs.append(scaled[i : i + self.seq_len])
			ys.append(scaled[i + self.seq_len])
		# 轉成 torch.Tensor
		self.x = torch.tensor(np.array(xs), dtype=self.dtype)        # [N, seq_len]
		self.y = torch.tensor(np.array(ys), dtype=self.dtype)        # [N]

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, idx):
		return self.x[idx].unsqueeze(-1), self.y[idx] # unsqueeze(-1) for LSTM-like models

	# 提供反縮放工具（可選）
	def inverse_transform_y(self, y_tensor: torch.Tensor):
		"""
		將標量或向量 y（縮放後）轉回原尺度。
		"""
		y_np = y_tensor.detach().cpu().numpy().reshape(-1, 1)
		return torch.tensor(self.scaler.inverse_transform(y_np).reshape(-1), dtype=y_tensor.dtype)

def make_bessel_dataset(data=j2, seq_len=4, batch_size=64, shuffle=True, feature_range=(-1, 1), dtype=torch.float32):
	"""
	方便的一次到位工廠：直接拿到 Dataset 與 DataLoader。
	"""
	ds = BesselSequenceDataset(data_src=data, seq_len=seq_len, feature_range=feature_range, dtype=dtype)
	dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

	return ds, dl


def plotting_test(data_src):
	ax1 = fig.add_subplot()
	ax1.plot(x,data_src)
	ax1.axhline(color="grey", ls="--", zorder=-1)
	ax1.set_ylim(-1,1)
	ax1.text(0.5, 0.95,'Bessel', ha='center', va='top',
		 transform = ax1.transAxes)

	plt.show()

def main():
	scaled_dataset = generate_dataset(j2)
	plotting_test(scaled_dataset)
	x, y = get_bessel_data(j2)
	print(x.shape)
	print(y.shape)

	# 2025 09 19 Test
	j2_dataset = BesselSequenceDataset(data_src = j2)
	# plotting_test(j2_dataset.y)
	print(j2_dataset.x.shape)
	print(j2_dataset.y.shape)

	ds, dl = make_bessel_dataset(data=j2, seq_len=4, batch_size=32, shuffle=False)
	for xb, yb in dl:
		print("x batch: {}".format(xb))
		print("y batch: {}".format(yb))
		print("x batch shape: {}".format(xb.shape))
		print("y batch shape: {}".format(yb.shape))


	

if __name__ == '__main__':
	main()
