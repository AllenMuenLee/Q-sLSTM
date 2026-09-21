
import copy 
import time
import os
import warnings
import torch.utils.data as Data
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
import math
import pennylane as qml

# Set random seed for reproducibility
seed = 93
np.random.seed(seed)
random.seed(seed)
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
torch.manual_seed(seed)


# Adjustable hyperparameters

hidden_layer_size = 200 # Set the hidden layer size here
lookback = 3
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
num_layers=3
num_blocks=10
dropout=0.005
hidden_sz =20
n_qubits =5
n_qlayers =2
input_size = lookback + 5
ts =0
tsp = 600


class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, input, hx):
        h, c = hx
        gates = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        
        i, f, g, o = gates.chunk(4, 1)
        
        i = torch.exp(i)  # Exponential input gate
        f = torch.exp(f)  # Exponential forget gate
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c = f * c + i * g
        h = o * torch.tanh(c)
        
        return h, c
    
class MQLSTM(nn.Module):
    def __init__(self, input_sz=input_size, hidden_sz=hidden_sz, n_qubits=n_qubits, n_qlayers=n_qlayers, batch_first=True,
                 return_sequences=False, return_state=False, backend="default.qubit",
                 four_Elayer_before_vqc=True, combine_Elayer_after_vqc=False,
                 find_unused_parameters=True, dropout=dropout):

        super(MQLSTM, self).__init__()

        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.concat_size = self.input_sz + self.hidden_sz
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend
        self.batch_first = batch_first
        self.dropout_rate = dropout
        self.return_sequences = return_sequences
        self.return_state = return_state
        # self.diff_method = self.diff_method

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        self.qlayer_forget = qml.QNode(
            self._circuit_forget, self.dev_forget, interface="torch")
        self.qlayer_input = qml.QNode(
            self._circuit_input, self.dev_input, interface="torch")
        self.qlayer_update = qml.QNode(
            self._circuit_update, self.dev_update, interface="torch")
        self.qlayer_output = qml.QNode(
            self._circuit_output, self.dev_output, interface="torch")

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(
            f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.VQC = {'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
                    'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
                    'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
                    'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
                    }

        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
        self.init_weights()

        self.four_Elayer_before_vqc = four_Elayer_before_vqc
        self.combine_Elayer_after_vqc= combine_Elayer_after_vqc
        # self.encoding = self.encoding

        # EMBEDDING
        self.dropout = torch.nn.Dropout(self.dropout_rate)

        # if not self.combine_linear_after_vqc:
        if self.combine_Elayer_after_vqc:
            self.Elayer_out = torch.nn.Linear(self.n_qubits, self.hidden_dim)

        else:
            self.Elayer_out_forget = torch.nn.Linear(
                self.n_qubits, self.hidden_sz)
            self.Elayer_out_input = torch.nn.Linear(
                self.n_qubits, self.hidden_sz)
            self.Elayer_out_update = torch.nn.Linear(
                self.n_qubits, self.hidden_sz)
            self.Elayer_out_output = torch.nn.Linear(
                self.n_qubits, self.hidden_sz)

        if self.four_Elayer_before_vqc:
            self.Elayer_in_forget = torch.nn.Linear(
                self.concat_size, self.n_qubits)
            self.Elayer_in_input = torch.nn.Linear(
                self.concat_size, self.n_qubits)
            self.Elayer_in_update = torch.nn.Linear(
                self.concat_size, self.n_qubits)
            self.Elayer_in_output = torch.nn.Linear(
                self.concat_size, self.n_qubits)
        else:
            self.Elayer_in = torch.nn.Linear(self.concat_size, self.n_qubits)

    # def _circuit_forget(self, inputs, weights):
    #         qml.templates.AmplitudeEmbedding(inputs, wires = self.wires_forget, pad_with=0.9, normalize=True)
    #         qml.templates.BasicEntanglerLayers(weights, wires = self.wires_forget)
    #         return [qml.expval(qml.PauliZ(wires = w)) for w in self.wires_forget]
    # def _circuit_input(self, inputs, weights):
    #         qml.templates.AmplitudeEmbedding(inputs, wires = self.wires_input, pad_with=0.9, normalize=True)
    #         qml.templates.BasicEntanglerLayers(weights, wires = self.wires_input)
    #         return [qml.expval(qml.PauliZ(wires = w)) for w in self.wires_input]
    # def _circuit_update(self, inputs, weights):

    #         qml.templates.AmplitudeEmbedding(inputs, wires=self.wires_update, pad_with=0.9, normalize=True)
    #         qml.templates.BasicEntanglerLayers(weights, wires = self.wires_update)
    #         return [qml.expval(qml.PauliZ(wires = w)) for w in self.wires_update]
    # def _circuit_output(self, inputs, weights):
    #         qml.templates.AmplitudeEmbedding(inputs, wires = self.wires_output,pad_with=0.9, normalize=True)
    #         qml.templates.BasicEntanglerLayers(weights, wires = self.wires_output)
    #         return [qml.expval(qml.PauliZ(wires = w)) for w in self.wires_output]

    def _circuit_forget(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]

    def _circuit_input(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]

    def _circuit_update(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]

    def _circuit_output(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        bs, seq_sz, _ = x.size()  #
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_sz),
                        torch.zeros(bs, self.hidden_sz))
        else:
            h_t, c_t = init_states
        for t in range(seq_sz):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)
        # match qubit dimension
        if self.four_Elayer_before_vqc:
            y_t_forget = self.Elayer_in_forget(v_t)
            y_t_input = self.Elayer_in_input(v_t)
            y_t_update = self.Elayer_in_update(v_t)
            y_t_output = self.Elayer_in_output(v_t)
            f_t_vqc = self.VQC['forget'](y_t_forget)  # forget block
            i_t_vqc = self.VQC['input'](y_t_input)
            g_t_vqc = self.VQC['update'](y_t_update)
            o_t_vqc = self.VQC['output'](y_t_output)
        else:
            y_t = self.Elayer_in(v_t)
            f_t_vqc = self.VQC['forget'](y_t)  # forget block
            i_t_vqc = self.VQC['input'](y_t)  # input block
            g_t_vqc = self.VQC['update'](y_t)  # update block
            o_t_vqc = self.VQC['output'](y_t)  # output block

        if self.combine_Elayer_after_vqc:
            f_t = torch.sigmoid(self.Elayer_out(f_t_vqc))
            i_t = torch.sigmoid(self.Elayer_out(i_t_vqc))
            g_t = torch.tanh(self.Elayer_out(g_t_vqc))
            o_t = torch.sigmoid(self.Elayer_out(o_t_vqc))
        else:
            f_t = torch.sigmoid(self.Elayer_out_forget(f_t_vqc))
            i_t = torch.sigmoid(self.Elayer_out_input(i_t_vqc))
            g_t = torch.tanh(self.Elayer_out_update(g_t_vqc))
            o_t = torch.sigmoid(self.Elayer_out_output(o_t_vqc))

        c_t = (f_t * c_t) + (i_t * g_t)
        h_t = o_t * torch.tanh(c_t)

        hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
    # hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=num_layers, num_blocks=num_blocks, dropout=dropout):
        """
        sLSTM layer with additional hyperparameters.
        
        Args:
            input_size (int): Size of input features.
            hidden_size (int): Size of hidden state.
            num_layers (int): Number of sLSTM layers.
            num_blocks (int): Number of blocks (cells) within each layer.
            dropout (float, optional): Dropout probability between layers.
        """
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks  # Additional hyperparameter
        self.dropout = dropout
        # Each layer has multiple blocks
        self.layers = nn.ModuleList([sLSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
                                     for i in range(num_layers)])
        self.lstm = MQLSTM()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_state=None):
        batch_size, seq_length, _ = input_seq.size()
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        
        outputs = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, c = hidden_state[layer_idx]
                h, c = layer(x, (h, c))
                hidden_state[layer_idx] = (h, c)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            outputs.append(x)
        
        return torch.stack(outputs, dim=1), hidden_state

    def init_hidden(self, batch_size):
        return [(torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device),
                 torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device))
                for _ in range(self.num_layers)]

class Model(nn.Module):
    def __init__(self, input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=1, 
                 num_layers=num_layers, num_blocks=num_blocks, dropout=dropout):
        """
        Model initialization with enhanced hyperparameters.

        Args:
            input_size (int): Input feature size.
            hidden_layer_size (int): Size of hidden layers.
            output_size (int): Output size.
            num_layers (int): Number of sLSTM layers.
            num_blocks (int): Number of blocks per layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks  # Additional block hyperparameter
        self.s_lstm = sLSTM(input_size=input_size, hidden_size=hidden_layer_size, num_layers=num_layers, num_blocks=num_blocks, dropout=dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        s_lstm_out, _ = self.s_lstm(x)
        predictions = self.linear(s_lstm_out[:, -1])
        return predictions

    def totolist1(self, x):
        total = []
        for i in range(len(x)):
            total.append(x[i].data.cpu().detach().numpy())      
        e1 = [token for st in total for token in st]
        return e1

def draw(preds):
    plt.figure(figsize=(7, 5))
    plt.plot(preds, color='royalblue')
    plt.legend(['Loss'], fontsize=15)
    plt.ylabel('Training loss', fontsize=15)
    plt.xlabel('Time (h)', fontsize=15)
    plt.show()

def loss(x, y):
    MSE = np.sum((x - y)**2) / len(y)
    RMSE = MSE ** 0.5
    MAE = np.sum(np.absolute(x - y)) / len(y)
    MAEP = np.sum(np.absolute(x - y) / y) / len(y) * 100
    return MSE, RMSE, MAE, MAEP

def lstm(input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=1, pre_model=None, freeze=False, verbose=True):
    model = Model(input_size, hidden_layer_size, output_size)
    if pre_model:
        pre_model_dict = pre_model.state_dict()
        if freeze:
            for param in pre_model_dict["linear.weight"]:
                param.requires_grad = True
    if verbose:
        print(model)
    return model

if __name__ == "__main__":
    

    file_path = os.path.join(os.getcwd(), "C:/Users/USER/Desktop/Q_sLSTM/VEH3.csv")
    battery = pd.read_csv(file_path)
    battery = battery.dropna()

    ts = 0
    data = battery.loc[:, ['charging_cycle', 'SOC_e', 'current', 'pack_voltage_e', 'Tmax', 'Tmin',
           'SOH']]
    data_train = data[ts:]

    # Data preparation
    data_cycle = pd.DataFrame(data_train['charging_cycle'])
    data_cycle_train = data_cycle
    
    data_HI = pd.DataFrame(data_train.iloc[:, 1:6])
    data_HI_train = data_HI
    
    data_TV = pd.DataFrame(data_train['SOH'])
    data_TV_train = data_TV

    cycle_sc, HI_sc, TV_sc = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    
    data_cycle_sc = cycle_sc.fit_transform(data_cycle)
    data_cycle_train_sc = cycle_sc.transform(data_cycle_train)
    
    data_HI_sc = HI_sc.fit_transform(data_HI)
    data_HI_train_sc = HI_sc.transform(data_HI_train)
    data_HI_train_sc_df = pd.DataFrame(data=data_HI_train_sc, columns=data_HI.columns)
    
    data_TV_sc = TV_sc.fit_transform(data_TV)
    data_TV_train_sc = TV_sc.transform(data_TV_train)
    data_TV_train_sc_df = pd.DataFrame(data=data_TV_train_sc, columns=['SOH'])

    for i in range(lookback):
        data_TV_train_sc_df['SOH' + str(i + 1)] = data_TV_train_sc_df['SOH'].shift(-i - 1)
        
    TV_first_column = data_TV_train_sc_df["SOH"]
    data_TV_train_sc_df = data_TV_train_sc_df.drop(["SOH"], axis=1)
    
    train_window = pd.concat([data_HI_train_sc_df[lookback:].reset_index(drop=True),
                              data_TV_train_sc_df],
                             axis=1, ignore_index=True)
        
    train_data = np.array(train_window)[:-lookback]
    
    # Ensure correct dimensions
    X_train = train_data[:, :] 
    y_train = np.array(TV_first_column.iloc[:-lookback])  # Target values
    
    # Reshape X_train
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
 
    if n_samples < lookback:
        raise ValueError("Number of samples is less than the lookback period. Please check your data preparation.")
    
    n_windows = n_samples - lookback + 1  # Number of sequences we can create
    
    # Create a new array to hold the reshaped data
    X_reshaped = np.zeros((n_windows, lookback, n_features))
    
    for i in range(n_windows):
        X_reshaped[i] = X_train[i:i + lookback]
    
    X_train = torch.FloatTensor(X_reshaped)  # Convert to tensor
    y_train = torch.FloatTensor(y_train[lookback - 1:])  # Adjust target variable to match new shape
    
    start = time.time()

    MQLSTM_model = lstm(input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=1, freeze=False, verbose=True)
  
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(MQLSTM_model.parameters(), lr=LEARNING_RATE)
    
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(X_train, y_train),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=0)

    MQLSTM_train_EPOCH_losses = []
 
    for i in range(EPOCHS):
        MQLSTM_model.train()
        MQLSTM_train_losses = []
        for MQLSTM_train_seq, MQLSTM_train_labels in train_loader:
            optimizer.zero_grad()
            MQLSTM_train_y_pred = MQLSTM_model(MQLSTM_train_seq)  # Get predictions
            MQLSTM_train_y_pred = MQLSTM_train_y_pred.squeeze(-1)  # Remove extra dimension
            loss_value = loss_function(MQLSTM_train_y_pred, MQLSTM_train_labels)
            loss_value.backward()
            optimizer.step()
            MQLSTM_train_losses.append(loss_value.item())
  
        MQLSTM_train_EPOCH_losses.append(np.mean(MQLSTM_train_losses))
  
        if (i + 1) % 1 == 0:
            print(f"Epoch {i + 1}/{EPOCHS}, Loss: {MQLSTM_train_EPOCH_losses[-1]:.4f}")

    # After training is complete, evaluate the model
    MQLSTM_model.eval()
    MQLSTM_valid_y_pred = MQLSTM_model(X_train).squeeze(-1)
    MQLSTM_valid_y_pred = MQLSTM_valid_y_pred.data.cpu().numpy().ravel()

    # Plot predictions and training loss after the last epoch
    plt.figure(num=1, figsize=(10, 6))
    plt.plot(np.arange(0, len(y_train)), y_train.data.cpu().numpy().ravel(), color="darkblue", label="y actual (train)")
    plt.plot(np.arange(0, len(y_train)), MQLSTM_valid_y_pred, color="darkred", label="y pred (train)")
    plt.grid(True)
    plt.legend(loc="best")
    plt.title("Training Predictions vs Actual")
    plt.xlabel("Samples")
    plt.ylabel("Value")
    plt.show()

    plt.figure(num=2, figsize=(10, 6))
    plt.plot(np.arange(0, len(MQLSTM_train_EPOCH_losses)), MQLSTM_train_EPOCH_losses, color="red", label="training loss")
    plt.grid(True)
    plt.legend(loc="best")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


torch.save(MQLSTM_model, "MQLSTM.pth")


### online rul prediction for target domain (only for one step-ahead prediction)
file_path = os.path.join(os.getcwd(
), "C:/Users/USER/Desktop/Q_sLSTM/VEH7.csv")
pre_model1 = torch.load("MQLSTM.pth")
battery1 = pd.read_csv(file_path)
battery1 = battery1.dropna()
original_battery = copy.copy(battery1)

seed = 93
np.random.seed(seed)
random.seed(seed)


n_output = 1

battery1 = battery1.loc[(battery1['charging_cycle'] >= ts)]

#### Data preparation

data2 = battery.loc[:, ['charging_cycle', 'SOC_e', 'current', 'pack_voltage_e', 'Tmax', 'Tmin',
       'SOH']]
data_train1 = data2[:tsp]
# Data preparation
data_cycle1 = pd.DataFrame(data_train1['charging_cycle'])
data_cycle_train1 = data_cycle1

data_HI1 = pd.DataFrame(data_train1.iloc[:, 1:6])
data_HI_train1 = data_HI1

data_TV1 = pd.DataFrame(data_train1['SOH'])
data_TV_train1 = data_TV1

cycle_sc, HI_sc, TV_sc = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

data_cycle_sc1 = cycle_sc.fit_transform(data_cycle1)
data_cycle_train_sc = cycle_sc.transform(data_cycle_train1)

data_HI_sc1 = HI_sc.fit_transform(data_HI1)
data_HI_train_sc1 = HI_sc.transform(data_HI_train1)
data_HI_train_sc_df1 = pd.DataFrame(data=data_HI_train_sc1, columns=data_HI1.columns)

data_TV_sc1 = TV_sc.fit_transform(data_TV1)
data_TV_train_sc1= TV_sc.transform(data_TV_train1)
data_TV_train_sc_df1 = pd.DataFrame(data=data_TV_train_sc1, columns=['SOH'])

for i in range(lookback):
    data_TV_train_sc_df1['SOH' + str(i + 1)] = data_TV_train_sc_df1['SOH'].shift(-i - 1) 
    
data_TV_train_sc_df2 = data_TV_train_sc_df1.drop(["SOH"], axis=1)  
TV_first_column1 = data_TV_train_sc_df1["SOH"]

train_window1 = pd.concat([data_HI_train_sc_df1[lookback:].reset_index(drop=True),
                         data_TV_train_sc_df2],
                         axis=1, ignore_index=True)
    
train_data1 = np.array(train_window1)[:-lookback]

# Ensure correct dimensions
X_train_T = train_data1[:, :] 
y_train_T = np.array(TV_first_column1.iloc[:-lookback])  # Target values

# Reshape X_train
n_samples_T = X_train_T.shape[0]
n_features_T = X_train_T.shape[1]
 
if n_samples_T < lookback:
    raise ValueError("Number of samples is less than the lookback period. Please check your data preparation.")

n_windows_T = n_samples_T - lookback + 1  # Number of sequences we can create

# Create a new array to hold the reshaped data
X_reshaped_T = np.zeros((n_windows_T, lookback, n_features_T))

for i in range(n_windows_T):
    X_reshaped_T[i] = X_train_T[i:i + lookback]

X_train_T = torch.FloatTensor(X_reshaped_T) 
y_train_T= torch.FloatTensor(y_train_T[lookback - 1:])  

## target model testing or FC2 stack votage prediction (SOH predciton )
data_test = data2[tsp:]
data_cycle2 = pd.DataFrame(data_test['charging_cycle'])
data_cycle_test = data_cycle2

data_HI2 = pd.DataFrame(data_test.iloc[:, 1:6])
data_HI_test = data_HI2

data_TV2 = pd.DataFrame(data_test['SOH'])
data_TV_test = data_TV2

cycle_sc, HI_sc, TV_sc = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

data_cycle_sc12 = cycle_sc.fit_transform(data_cycle2)
data_cycle_test_sc = cycle_sc.transform(data_cycle_test)

data_HI_sc2 = HI_sc.fit_transform(data_HI2)
data_HI_test_sc = HI_sc.transform(data_HI_test)
data_HI_test_sc_df = pd.DataFrame(data=data_HI_test_sc, columns=data_HI2.columns)

data_TV_sc2 = TV_sc.fit_transform(data_TV2)
data_TV_test_sc= TV_sc.transform(data_TV_test)
data_TV_test_sc_df = pd.DataFrame(data=data_TV_test_sc, columns=['SOH'])

for i in range(lookback):
   data_TV_test_sc_df['SOH'+str(i+1)] = data_TV_test_sc_df['SOH'].shift(-i-1)
data_TV_test_sc_df1= data_TV_test_sc_df.drop(["SOH"], axis= 1)
TV_first_column3 = data_TV_test_sc_df["SOH"]

test_window = pd.concat([data_HI_test_sc_df,
                           data_TV_test_sc_df1], axis=1,ignore_index=False)

test_data = np.array(test_window)[:-lookback]

X_test_T, y_test_T,= test_data, np.array(TV_first_column3[lookback:]).reshape(-1, 1)

global result
result = '\nEvaluation.'
result1= 'Computation Time'

start = time.time()
# Reshape X_train
n_samples_test = X_test_T.shape[0]
n_features_test = X_test_T.shape[1]
 
if n_samples_test < lookback:
    raise ValueError("Number of samples is less than the lookback period. Please check your data preparation.")

n_windows_test = n_samples_test - lookback + 1  # Number of sequences we can create

# Create a new array to hold the reshaped data
X_reshaped_test = np.zeros((n_windows_test, lookback, n_features_test))

for i in range(n_windows_test):
    X_reshaped_test[i] = X_test_T[i:i + lookback]

X_test_T= torch.FloatTensor(X_reshaped_test)  # Convert to tensor
y_test_T = torch.FloatTensor(y_test_T[lookback - 1:])  # Adjust target variable to match new shape



def lstm_TL(input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=1, pre_model=pre_model1, freeze=True, verbose=True):
    model = lstm(input_size, hidden_layer_size, output_size)
    if pre_model:
        pre_model_dict = pre_model.state_dict()
        if freeze:
            for param in pre_model_dict["linear.weight"]:
                param.requires_grad= True
    if verbose:
        print(model)
    return model

MQLSTM_model_TR  = lstm_TL(input_size=input_size, hidden_layer_size= hidden_layer_size, output_size=1, pre_model=pre_model1, freeze=True)
optimizer = torch.optim.Adam(MQLSTM_model_TR.parameters(), lr = LEARNING_RATE)
loss_function = nn.MSELoss()


train_loader_T = Data.DataLoader(dataset = Data.TensorDataset(X_train_T,y_train_T), 
                                                    batch_size = BATCH_SIZE,                                                    
                                                num_workers = 0)

MQLSTM_test_EPOCH_losses  = [] 
 
for i in range(EPOCHS):
    MQLSTM_model_TR.train()
    MQLSTM_test_losses  = []
    for MQLSTM_test_seq, MQLSTM_test_labels in train_loader_T:
        optimizer.zero_grad()
        MQLSTM_test_y_pred = MQLSTM_model_TR(MQLSTM_test_seq)  # Get predictions
        MQLSTM_test_y_pred = MQLSTM_test_y_pred.squeeze(-1)  # Remove extra dimension
        loss_value = loss_function(MQLSTM_test_y_pred, MQLSTM_test_labels)
        loss_value.backward()
        optimizer.step()
        MQLSTM_test_losses.append(loss_value.item())
  
    MQLSTM_test_EPOCH_losses.append(np.mean(MQLSTM_test_losses))
  
    if (i + 1) % 1 == 0:
        print(f"Epoch {i + 1}/{EPOCHS}, Loss: {MQLSTM_test_EPOCH_losses[-1]:.4f}")
end = time.time()
result1=end-start

              
MQLSTM_model_TR.eval()
y_pred_train_T  = MQLSTM_model_TR(X_train_T).squeeze(-1)
y_pred_train_T = y_pred_train_T.detach().numpy() 
y_pred_train_T = TV_sc.inverse_transform(y_pred_train_T.reshape(1,-1))
y_train_T   =    TV_sc.inverse_transform(y_train_T.reshape(1,-1))
y_train1 = pd.DataFrame(y_train_T.reshape(-1, 1))
y_pred_train_T1 = pd.DataFrame(y_pred_train_T.reshape(-1, 1))


plt.figure(num= 1, figsize= (10, 6))
plt.plot(np.arange(0, len(y_pred_train_T1)), y_train1, color= "darkblue", label= "y actual (train)")
plt.plot(np.arange (0, len(y_pred_train_T1)), y_pred_train_T1, color= "darkred", label= "y pred (train)")
plt.grid(True)
plt.legend(loc= "best")
plt.show()

#testing
y_pred_test_T  = MQLSTM_model_TR(X_test_T)
y_pred_test_T = y_pred_test_T.detach().numpy() 
y_pred_test_T =TV_sc.inverse_transform(y_pred_test_T.reshape(1,-1))
y_test_T=      TV_sc.inverse_transform(y_test_T.reshape(1,-1))
y_test1 = pd.DataFrame(y_test_T.reshape(-1, 1))
y_pred_test_T1 = pd.DataFrame(y_pred_test_T.reshape(-1, 1))
# pd.DataFrame(y_pred_test_T.reshape(-1, 1)).to_csv("C:/Users/MA201_5950X/Desktop/sample codes of QLSTM2/voltage pred_21324/MQLSTM/vltg_pred_621/2_5_611_39.csv")


plt.figure(num= 1, figsize= (10, 6))
plt.plot(np.arange(0, len(y_pred_test_T1)), y_test1, color= "darkblue", label= "y actual (train)")
plt.plot(np.arange (0, len(y_pred_test_T1)), y_pred_test_T1, color= "darkred", label= "y pred (train)")
plt.grid(True)
plt.legend(loc= "best")
plt.show()

MSE,RMSE,MAE, MAEP= loss(y_pred_test_T1,  y_test1)
# result1+= '\nRunning Time LSTM: {}'.format(end2-start2)
result += '\n\nMQLSTM_MAE: {}'.format(MAE)
result += '\nMQLSTM_RMSE: {}'.format(RMSE)
result += '\nMQLSTM_MSE: {}'.format(MSE)
result += '\nMQLSTM_MAEPE: {}'.format(MAEP)

print(result)
print(f"Training completed in {round((result1), 2)} seconds")       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
