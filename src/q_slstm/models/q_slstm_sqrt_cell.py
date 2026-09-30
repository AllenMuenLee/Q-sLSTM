# Q-sLSTM variant with square-root input and forget gates: the stabilized (h, c, n, m) cell of
# q_slstm_cell.py with
#     i = sqrt((1 + q_i) / (1 - q_i)),  f = sqrt((1 + q_f) / (1 - q_f)),  o = sigmoid(q_o)
# Both gates are kept in the log domain (half the clamped log ratio) for the xLSTM stabilizer.

from .q_slstm_cell import DEFAULT_GATE_EPSILON, CustomQsLSTMCell, bounded_log_ratio

QSLSTM_SQRT_RECURRENCE = "xlstm_stabilized_sqrt_input_forget_v1"


class CustomQsLSTMSqrtCell(CustomQsLSTMCell):
    """Q-sLSTM cell with sqrt((1 + q) / (1 - q)) input and forget gates; output gate, VQCs, state
    (h, c, n, m) and construction order are inherited, so equal seeds give equal initial parameters."""

    def __init__(self, input_size, hidden_size, output_size, vqc_depth, gate_epsilon=DEFAULT_GATE_EPSILON):
        super().__init__(input_size, hidden_size, output_size, vqc_depth, gate_epsilon)
        self.recurrence = QSLSTM_SQRT_RECURRENCE

    def log_input(self, q_i):
        """Log of sqrt((1 + q) / (1 - q))."""
        return 0.5 * bounded_log_ratio(q_i, self.gate_epsilon)

    def log_forget(self, q_f):
        """Log of sqrt((1 + q) / (1 - q))."""
        return 0.5 * bounded_log_ratio(q_f, self.gate_epsilon)
