# Reference ("fk") Q-sLSTM cell, ported from src/fk_q_slstm/QsLSTM_102124.py.
#
# The fk QLSTM cell (fk_qlstm_cell.py: classical encoders around every gate VQC, unnormalized (h, c)
# recurrence) with the reference sLSTMCell's exponential input and forget gates:
#     i = exp(.), f = exp(.), g = tanh(.), o = sigmoid(.)
#     c_t = f * c_prev + i * g,  h_t = o * tanh(c_t)
# No normalizer and no stabilizer: c_t can grow geometrically; a non-finite state is caught by the
# trainer's finiteness checks and recorded as a failed run.

import torch

from .fk_qlstm_cell import CustomFkQLSTMCell

FK_QSLSTM_RECURRENCE = "fk_exp_gates_classical_encoders_v1"


class CustomFkQsLSTMCell(CustomFkQLSTMCell):
    """Reference Q-sLSTM cell (exponential input/forget gates) with recurrent state (h, c).

    Same construction order and parameter names as CustomFkQLSTMCell, so equal seeds give identical
    initial parameters and the two differ only in the input/forget activations.
    """

    input_activation = forget_activation = staticmethod(torch.exp)
    recurrence = FK_QSLSTM_RECURRENCE
