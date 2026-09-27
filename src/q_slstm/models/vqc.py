# Variational quantum circuit shared by the QLSTM baseline and the Q-sLSTM cell.

import pennylane as qml
import torch
import torch.nn as nn

from .vqc_components import H_layer
from .vqc_components import RY_layer
from .vqc_components import entangling_layer

torch.set_default_dtype(torch.float32)


# Define actual circuit architecture
def q_function(x, q_weights, n_class):
    """ The variational quantum circuit. """

    n_dep = q_weights.shape[0]
    n_qub = q_weights.shape[1]

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qub)

    # Embed features in the quantum node
    RY_layer(x)

    # Sequence of trainable variational layers
    for k in range(n_dep):
        entangling_layer(n_qub)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    # only measure first "n_class" of qubits and discard the rest
    return [qml.expval(qml.PauliZ(position)) for position in range(n_class)]


class VQC(nn.Module):
    def __init__(self, vqc_depth, n_qubits, n_class):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))  # rotation params
        self.dev = qml.device("default.qubit", wires=n_qubits)  # Can use different simulation backend or quantum computers.

        self.VQC = qml.QNode(q_function, self.dev, interface="torch")

        self.n_qubits = n_qubits
        self.n_class = n_class

    def forward(self, X):
        # PennyLane parameter broadcasting: X [batch, n_qubits] -> n_class tensors of [batch].
        expvals = torch.stack(self.VQC(X, self.weights, self.n_class))
        # The simulator may promote to float64; hand back the caller's floating dtype.
        y_preds = expvals.movedim(0, -1).to(X.dtype)  # [n_class, batch] -> [batch, n_class]

        if X.ndim != 2 or y_preds.shape != (X.shape[0], self.n_class):
            raise ValueError(
                f"VQC expected input [batch, {self.n_qubits}] and output [batch, {self.n_class}], "
                f"got input {tuple(X.shape)} and output {tuple(y_preds.shape)}"
            )
        return y_preds
