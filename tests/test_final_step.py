import csv
from argparse import Namespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from q_slstm.trainers.train import _extract_model_output
from q_slstm.trainers.utils import predict_and_log

BATCH, SEQ, OUT = 4, 3, 2


def make_outputs():
    return torch.arange(BATCH * SEQ * OUT, dtype=torch.float32).reshape(BATCH, SEQ, OUT)


@pytest.mark.parametrize("model_name", ["qlstm", "qslstm", "lstm"])
def test_training_selects_last_step(model_name):
    outputs = make_outputs()
    state = (torch.zeros(BATCH, 1),) * (4 if model_name == "qslstm" else 2)

    prediction = _extract_model_output(Namespace(model=model_name), (outputs, state))

    assert prediction.shape == (BATCH, OUT)
    assert torch.equal(prediction, outputs[:, -1, :])
    # explicit values: row b, last step, coordinate k
    assert prediction[1, 0].item() == 1 * SEQ * OUT + (SEQ - 1) * OUT + 0
    assert prediction[3, 1].item() == 3 * SEQ * OUT + (SEQ - 1) * OUT + 1


class EchoModel(nn.Module):
    """Treats the input as already being the per-step outputs [batch, sequence, output]."""

    def forward(self, x):
        return x, ()


@pytest.mark.parametrize("batch_size", [2, 4])
def test_prediction_logging_selects_last_step(tmp_path, batch_size):
    outputs = make_outputs()
    targets = outputs[:, -1, :].clone()
    loader = DataLoader(TensorDataset(outputs, targets), batch_size=batch_size, shuffle=False)
    csv_path = tmp_path / "prediction_log.csv"
    args = Namespace(model="qlstm", device="cpu", horizon=1)

    predict_and_log(args, EchoModel(), loader, train_len=0, csv_path=csv_path, epoch=1, debug_plot=False)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == BATCH * OUT
    for row in rows:
        assert float(row["y_pred"]) == float(row["y_true"])
    logged = torch.tensor([float(r["y_pred"]) for r in rows]).reshape(BATCH, OUT)
    assert torch.equal(logged, outputs[:, -1, :])
