# VQC building blocks (gate layers) used by the QLSTM circuit

# Datetime
from datetime import datetime
import time


import matplotlib.pyplot as plt
from pandas import DataFrame

import warnings


import torch
import torch.nn as nn
import torch.optim as optim 

import pennylane as qml
from pennylane import numpy as np

from functools import *

# Saving
import pickle
import os
import copy




### VQC

def H_layer(nqubits):
	"""Layer of single-qubit Hadamard gates.
	"""
	for idx in range(nqubits):
		qml.Hadamard(wires=idx)

def RX_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx in range(w.shape[-1]):
		qml.RX(w[..., idx], wires=idx)

def RY_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx in range(w.shape[-1]):
		qml.RY(w[..., idx], wires=idx)

def RZ_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx in range(w.shape[-1]):
		qml.RZ(w[..., idx], wires=idx)


def entangling_layer(nqubits):
	"""Layer of CNOTs followed by another shifted layer of CNOT.
	"""
	# In other words it should apply something like :
	# CNOT  CNOT  CNOT  CNOT...  CNOT
	#   CNOT  CNOT  CNOT...  CNOT
	for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
		qml.CNOT(wires=[i, i + 1])
	for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
		qml.CNOT(wires=[i, i + 1])
	

def cycle_entangling_layer(nqubits):
	for i in range(0, nqubits):  
		qml.CNOT(wires=[i, (i + 1) % nqubits])
		
