# From https://skill-lync.com/projects/Solving-2nd-order-ODE-for-a-simple-pendulum-using-python-40080

import numpy as np
from scipy.integrate import odeint
import math
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

def system(theta,t,b,g,l,m):
	theta1 = theta[0]
	theta2 = theta[1]
	dtheta1_dt = theta2
	dtheta2_dt = -(b/m)*theta2-g*math.sin(theta1)
	dtheta_dt=[dtheta1_dt,dtheta2_dt]

	return dtheta_dt



b=0.15
g=9.81
l=1
m=1


theta_0 = [0,3]


t = np.linspace(0,20,240)


theta = odeint(system,theta_0,t,args = (b,g,l,m))



# f=1
# for i in range(0,240):
# 	filename = str(f)+'.png'
# 	f= f+1
# 	plt.figure()
# 	plt.plot([10,l*math.sin(theta[i,0])+10],[10,10-l*math.cos(theta[i,0])],marker="o")
# 	plt.xlim([0,20])
# 	plt.ylim([0,20])

	
# 	plt.savefig(filename)
	
# Plotting	
# plt.plot(t,theta[:,0],'b-')
# plt.plot(t,theta[:,1],'r--')
# plt.show()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(theta[:,1].reshape(-1, 1))


def plotting_test(x, data_src):
	# create a figure window
	fig = plt.figure(1, figsize=(9,8))
	ax1 = fig.add_subplot()
	ax1.plot(x,data_src)
	ax1.axhline(color="grey", ls="--", zorder=-1)
	ax1.set_ylim(-1,1)
	ax1.text(0.5, 0.95,'Damped SHM', ha='center', va='top',
		 transform = ax1.transAxes)

	plt.show()

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



def get_damped_shm_data(data = dataset, seq_len = 4):
	return transform_data_single_predict(data = data, seq_length = seq_len)


# 2025 09 19: Dataset

class DampedSHMSequenceDataset(Dataset):
	"""
	將一條 time-series（例如 theta[:,1] 或已縮放的 dataset）變成
	以連續 seq_len 點預測下一點的 (x, y) 樣本集合。

	預設假設 data_src 已經縮放好（pre_scaled=True），
	若要讓此 Dataset 內部幫你縮放，請把 pre_scaled=False。
	"""
	def __init__(self,
				 data_src=dataset,         # 預設用你已經算好的全域變數 `dataset`
				 seq_len=4,
				 pre_scaled=True,
				 feature_range=(-1, 1),
				 dtype=torch.float32):
		self.seq_len = int(seq_len)
		self.dtype = dtype

		arr = np.asarray(data_src).reshape(-1)

		# 是否要在 Dataset 內做縮放
		self.scaler = None
		if pre_scaled:
			scaled = arr
		else:
			self.scaler = MinMaxScaler(feature_range=feature_range)
			scaled = self.scaler.fit_transform(arr.reshape(-1, 1)).reshape(-1)

		# 預先切好所有 (x, y)
		xs, ys = [], []
		for i in range(len(scaled) - self.seq_len - 1):
			xs.append(scaled[i : i + self.seq_len])
			ys.append(scaled[i + self.seq_len])

		self.x = torch.tensor(np.array(xs), dtype=self.dtype)   # [N, seq_len]
		self.y = torch.tensor(np.array(ys), dtype=self.dtype)   # [N]

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, idx):
		# return self.x[idx], self.y[idx]
		return self.x[idx].unsqueeze(-1), self.y[idx] # unsqueeze(-1) for LSTM-like models

	def inverse_transform_y(self, y_tensor: torch.Tensor):
		"""
		若本 Dataset 內部有做縮放（pre_scaled=False），可用這個把輸出反縮放回原單位。
		若 pre_scaled=True（預設），此函數會直接回傳輸入。
		"""
		if self.scaler is None:
			return y_tensor
		y_np = y_tensor.detach().cpu().numpy().reshape(-1, 1)
		inv = self.scaler.inverse_transform(y_np).reshape(-1)
		return torch.tensor(inv, dtype=y_tensor.dtype)


def make_damped_shm_dataset(data_src=dataset,
							seq_len=4,
							batch_size=32,
							shuffle=True,
							pre_scaled=True,
							feature_range=(-1, 1),
							dtype=torch.float32):
	"""
	工廠函式：一次拿到 Dataset 與 DataLoader。
	預設使用全域變數 `dataset`（已縮放），若傳入未縮放序列，把 pre_scaled=False。
	"""
	ds = DampedSHMSequenceDataset(
		data_src=data_src,
		seq_len=seq_len,
		pre_scaled=pre_scaled,
		feature_range=feature_range,
		dtype=dtype
	)
	dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
	return ds, dl

def main():
	x, y = get_damped_shm_data()
	plotting_test(t, dataset)

	print(x.size())
	print(y.size())

	full_ds = DampedSHMSequenceDataset()

	print(full_ds.x.size())
	print(full_ds.y.size())

	plotting_test(t[5:], full_ds.y)





if __name__ == '__main__':
	main()