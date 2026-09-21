
import copy
import time
import os
import warnings
import torch.utils.data as Data
import math
import pennylane as qml
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
seed = 93
np.random.seed(seed)
random.seed(seed)
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
seed = 123
seed = 93
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class MQLSTM(nn.Module):
    def __init__(self, input_sz=21, hidden_sz=5, n_qubits=5, n_qlayers=2, batch_first=True,
                 return_sequences=False, return_state=False, backend="default.qubit",
                 four_Elayer_before_vqc=True, combine_Elayer_after_vqc=False,
                 find_unused_parameters=True, dropout=0.2):

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


class LSTM(nn.Module):
    def __init__(self, input_size=21, hidden_layer_size=5, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = MQLSTM()
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        lstm_out, (h_n, h_c) = self.lstm(input_x)
        lstm_out = self.sigmoid(lstm_out)
        predictions = self.linear(lstm_out.view(len(input_x), -1))

        return predictions
    def totolist1(self, x):
        total = []
        for i in range(len(x)):
            total.append(x[i].data.cpu().detach().numpy())      
        e1 = [token for st in total for token in st]
        return(e1)

def draw(preds):
    plt.figure(figsize=(7, 5))
    plt.plot(preds, color='royalblue')
    # plt.plot(preds1,color='green')
    # plt.plot(true,color='crimson')
    plt.legend(['Loss'], fontsize=15)
    plt.ylabel('Training loss', fontsize=15)
    plt.xlabel('Time (h)', fontsize=15)
    # plt.title('Predicted Vs Actual Voltage plot', fontsize=15)
    plt.show()

def draw2(preds, preds1, true):
    plt.figure(figsize=(7, 5))
    plt.plot(preds, color='royalblue')
    plt.plot(preds1, color='green')
    plt.plot(true, color='crimson')
    plt.legend(['prediction', 'actual'], fontsize=15)
    plt.axhline(y=0.8, color='rosybrown', linestyle='-')
    plt.ylabel('SOH', fontsize=15)
    plt.xlabel('Cycle', fontsize=15)
    plt.title('Predicted Vs Actual SOH plot for test data', fontsize=15)
    plt.show()

def loss(x, y):  # x: predicted value, y:actual
    MSE = np.sum((x-y)**2)/len(y)
    RMSE = MSE ** 0.5
    MAE = np.sum(np.absolute(x-y))/len(y)
    MAEP = np.sum(np.absolute(x-y)/y)/len(y)*100

    return MSE, RMSE, MAE, MAEP


# def lstm(input_size=21, hidden_layer_size=5, output_size=1, pre_model=None, freeze=False, verbose=True):

#     model = LSTM(input_size=21, hidden_layer_size=5, output_size=1)
#     model_children = list(model.children())
#     # pre_model_children = list(pre_model.children())
    
#     for i in range(2, len(model_children ) - 1):
#         if type(model_children [i]) :
           
#             # model_children[i].load_state_dict(pre_model_children[i].state_dict())
            
#             if freeze:
#                 for param in model_children[i].parameters():
#                     param.requires_grad = False
#         else:
#             print(f"Skipping layer {i} due to type mismatch.")
#           # model[i].set_weights(pre_model[i].get_weights())
#           # if freeze: 
#               # model[i].trainable = False
     
#     # model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
#     # if verbose: print(model)    
    
#     return model

def lstm(input_size=21, hidden_layer_size=5, output_size=1, pre_model=None, freeze=False, verbose=True):
    model = LSTM(input_size, hidden_layer_size, output_size)
    if pre_model:
          pre_model_dict = pre_model.state_dict()
          if freeze:
              for param in pre_model_dict["linear.weight"]:
                  param.requires_grad= True
    if verbose:
        print(model)   
    
    return model

#Loading the source dataset 

if __name__== "__main__":
    file_path= os.path.join(os.getcwd(),"C:/Users/USER/Desktop/second paper/MODEL/FC1_features_original.csv")
    battery= pd.read_csv(file_path)
    battery= battery.dropna()
    # print(battery.isnull().sum())
    
## Setting the experimental enviroment and hyperparametors 
    ts= 335
    n_output= 1
    lookback = 10
    
    #Expt1:5Q with 2 Qlayers
    BATCH_SIZE= 20
    EPOCHS=500
    LEARNING_RATE= 0.001
    
    #Expt2:5Q with 3 Qlayers
    # BATCH_SIZE= 35
    # EPOCHS=350
    # LEARNING_RATE= 0.001
    
    # #Expt3:5Q with 4 Qlayers
    # BATCH_SIZE= 35
    # EPOCHS=350
    # LEARNING_RATE= 0.001
    
      
    # #Expt4:7Q with 3 Qlayers
    # BATCH_SIZE= 35
    # EPOCHS=1000
    # LEARNING_RATE= 0.001
    
    # #Expt5:9Q with 3 Qlayers
    # BATCH_SIZE= 35
    # EPOCHS=1000
    # LEARNING_RATE= 0.001

    
    data= battery.loc[:, ['Time','V1','V2','V3','V4','V5','ToutH2','TinAIR','ToutAIR','TinWAT','DoutH2','DoutAIR', 'TV']]
    
    data_cycle= pd.DataFrame(data['Time'])
    data_cycle_train= data_cycle[ts:]
    data_HI= pd.DataFrame(data.iloc[:,1:12])
    data_HI_train= data_HI[ts:]
    data_SOH= pd.DataFrame(data['TV'])
    data_SOH_train= data_SOH[ts: ]
    
    # data_train= data[ts:]
    # Tl=int(len(data_train)*0.65) ##eol=2235
    # data_train_Tl= data_train[:Tl]
    # data_cycle= pd.DataFrame(data_train_Tl['Time'])
    # data_cycle_train=data_cycle
    # data_HI= pd.DataFrame(data_train_Tl.iloc[:,1:12])
    # data_HI_train= data_HI
    # data_SOH= pd.DataFrame(data_train_Tl['TV'])
    # data_SOH_train=data_SOH

    cycle_sc, HI_sc, TV_sc= MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    data_cycle_sc= cycle_sc.fit_transform(data_cycle)
    data_cycle_sc_df= pd.DataFrame(data=data_cycle_sc, columns=['Time'])
    data_cycle_train_sc= cycle_sc.transform(data_cycle_train)
    data_cycle_train_sc_df= pd.DataFrame(data=data_cycle_train_sc, columns=['Time'])
    data_HI_sc = HI_sc.fit_transform(data_HI)
    data_HI_train_sc = HI_sc.transform(data_HI_train)
    data_HI_train_sc_df = pd.DataFrame(data=data_HI_train_sc, columns= data_HI.columns)
    data_TV_sc = TV_sc.fit_transform(data_SOH)
    data_TV_train_sc = TV_sc.transform(data_SOH_train)
    data_TV_train_sc_df = pd.DataFrame(data=data_TV_train_sc, columns=['TV'])
    
    for i in range(lookback):
        data_TV_train_sc_df ['TV'+ str(i+1)]= data_TV_train_sc_df ['TV'].shift(-i- 1)
        
    TV_first_column = data_TV_train_sc_df["TV"]
    data_TV_train_sc_df = data_TV_train_sc_df.drop(["TV"], axis= 1)
    
    train_window = pd.concat([data_HI_train_sc_df[lookback:].reset_index(drop= True),
                                                data_TV_train_sc_df],
                                                axis= 1,
                                                ignore_index= True)
    # train_window = pd.concat([data_HI_train_sc_df.reset_index(drop= True),
    #                                             data_TV_train_sc_df],
    #                                             axis= 1,
    #                                             ignore_index= True)
        
    train_data= np.array(train_window)[: -lookback]
    
    X_train, y_train,= train_data, np.array(TV_first_column.iloc[lookback:])
    
    
    X_roll= X_train.copy()
    y_roll= y_train.copy()
    X_train= X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_train, y_train= torch.FloatTensor(X_train), torch.FloatTensor(y_train)


    
#### Training the source model 
    
    start= time.time()

    # MQLSTM_model  = lstm(input_size=21, hidden_layer_size=5, output_size=1, pre_model=None, freeze=False, verbose=True)
    
    MQLSTM_model  = lstm(input_size=21, hidden_layer_size=5, output_size=1, pre_model=None, freeze=False, verbose=True)
    
    # MQLSTM_model= LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(MQLSTM_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.1, patience=5, verbose=True)   
    epochs = EPOCHS
    # train data
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(X_train, y_train),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=0,
                                   )

    MQLSTM_history = {'QLSTM_train_Loss':    []}

    MQLSTM_train_EPOCH_losses = []

    for i in range(epochs):
        MQLSTM_model.train()
        MQLSTM_train_losses = []
        MQLSTM_train_preds = []
        MQLSTM_train_targets = []
        for MQLSTM_train_seq, MQLSTM_train_labels in train_loader:
            optimizer.zero_grad()
            MQLSTM_train_y_pred = MQLSTM_model(MQLSTM_train_seq).squeeze(-1)
            # QLSTM_train_y_pred = np.squeeze(model(QLSTM_train_seq), -1).detach().numpy()
            MQLSTM_train_single_loss = loss_function(
                MQLSTM_train_y_pred, MQLSTM_train_labels)

            MQLSTM_train_single_loss.backward()
            optimizer.step()

            MQLSTM_train_preds.append(MQLSTM_train_y_pred)
            MQLSTM_train_targets.append(MQLSTM_train_labels)
            MQLSTM_train_losses.append(float(MQLSTM_train_single_loss))
            # print("Train Step:", i, " QLSTM_train_loss: ", QLSTM_train_single_loss)
        MQLSTM_train_preds = [round(r)
                             for r in MQLSTM_model.totolist1(MQLSTM_train_preds)]
        MQLSTM_train_targets = MQLSTM_model.totolist1(MQLSTM_train_targets)
        MQLSTM_train_avg_loss = np.mean(MQLSTM_train_losses)
        scheduler.step(MQLSTM_train_avg_loss)
        print(
            f"Epoch {i} :  MQLSTM_train_avg_Loss = {MQLSTM_train_avg_loss:2.6f} ")
        
        MQLSTM_train_EPOCH_losses.append(MQLSTM_train_avg_loss)
        del MQLSTM_train_preds, MQLSTM_train_targets

    MQLSTM_model.eval()
    MQLSTM_valid_y_pred = MQLSTM_model(X_train).squeeze(-1)
    MQLSTM_valid_y_pred = MQLSTM_valid_y_pred.data.cpu().numpy().ravel()
    
    end= time.time()
    print(round((end-start), 2),"seconds")


    plt.figure(num=1, figsize=(10, 6))
    plt.plot(np.arange(0, len(y_train)), y_train.data.cpu(
    ).numpy().ravel(), color="darkblue", label="y actual (train)")
    plt.plot(np.arange(0, len(y_train)), MQLSTM_valid_y_pred,
             color="darkred", label="y pred (train)")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

    MQLSTM_train_EPOCH_losses1 = pd.DataFrame(
        np.array(MQLSTM_train_EPOCH_losses).reshape(-1, 1))
    # MQLSTM_train_EPOCH_losses1 .to_csv("C:/Users/MA201_5950X/Desktop/sample codes of QLSTM2/voltage pred_21324/MQLSTM/vltg_pred_621/Training_loss3_9_533.csv")
    plt.figure(num=1, figsize=(10, 6))
    plt.plot(np.arange(0, len(MQLSTM_train_EPOCH_losses)),
             MQLSTM_train_EPOCH_losses, color="red", label="training lsoss")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()


torch.save(MQLSTM_model, "MQLSTM.pth")


### online rul prediction for target domain (only for one step-ahead prediction)
file_path = os.path.join(os.getcwd(
), "C:/Users/USER/Desktop/second paper/MODEL/FC2_features_original.csv")
pre_model1 = torch.load("MQLSTM.pth")
battery1 = pd.read_csv(file_path)
battery1 = battery1.dropna()
original_battery = copy.copy(battery1)
seed = 123
seed = 93
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

ts = 258
tsp = 711
n_output = 1
lookback = 10

#Expt1:5Q with 2 Qlayers
BATCH_SIZE= 10
EPOCHS=200
LEARNING_RATE= 0.0021

# #Expt2:5Q with 3 Qlayers
# BATCH_SIZE= 10
# EPOCHS = 50
# LEARNING_RATE = 0.015

# #Expt3:5Q with 4 Qlayers
# BATCH_SIZE= 25
# EPOCHS = 200
# LEARNING_RATE = 0.0189

  
# #Expt4:7Q with 3 Qlayers
# BATCH_SIZE= 10
# EPOCHS = 100
# LEARNING_RATE = 0.015

# #Expt5:9Q with 3 Qlayers
# BATCH_SIZE= 10
# EPOCHS = 50
# LEARNING_RATE = 0.015

# # best
# BATCH_SIZE= 10
# EPOCHS = 50
# LEARNING_RATE = 0.015

# # BATCH_SIZE=30
# # EPOCHS = 100
# # LEARNING_RATE = 0.01

battery1 = battery1.loc[(battery1["Time"] >= ts)]

data1= battery1.loc[:, ['Time','V1','V2','V3','V4','V5','ToutH2','TinAIR','ToutAIR','TinWAT','DoutH2','DoutAIR', 'TV']]


data_cycle1= pd.DataFrame(data1['Time'])
data_HI1= pd.DataFrame(data1.iloc[:,1:12])
data_SOH1= pd.DataFrame(data1['TV'])

###training 
data_cycle_train1= data_cycle1.loc[:tsp]
data_HI_train1= data_HI1.loc[:tsp]
data_SOH_train1= data_SOH1.loc[: tsp]

###testing 
data_cycle_test1= data_cycle1.loc[tsp:]
data_HI_test1= data_HI1.loc[tsp:]
data_SOH_test1= data_SOH1.loc[tsp:]

### trainign data normalization 
cycle_sc1, HI_sc1, TV_sc1= MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

data_cycle_sc1= cycle_sc1.fit_transform(data_cycle1)
data_cycle_sc_df1= pd.DataFrame(data=data_cycle_sc1, columns=['Time'])
data_cycle_train_sc1= cycle_sc1.transform(data_cycle_train1)
data_cycle_train_sc_df1= pd.DataFrame(data=data_cycle_train_sc1, columns=['Time'])

data_HI_sc1 = HI_sc1.fit_transform(data_HI1)
data_HI_train_sc1 = HI_sc1.transform(data_HI_train1)
data_HI_train_sc_df1 = pd.DataFrame(data=data_HI_train_sc1, columns= data_HI1.columns)

data_TV_sc1 = TV_sc1.fit_transform(data_SOH1)
data_TV_train_sc1 = TV_sc1.transform(data_SOH_train1)
data_TV_train_sc_df1 = pd.DataFrame(data=data_TV_train_sc1, columns=['TV'])

for i in range(lookback):
    data_TV_train_sc_df1 ['TV'+ str(i+1)]= data_TV_train_sc_df1 ['TV'].shift(-i- 1)
    
TV_first_column1 = data_TV_train_sc_df1["TV"]
data_TV_train_sc_df1 = data_TV_train_sc_df1.drop(["TV"], axis= 1)

train_window1 = pd.concat([data_HI_train_sc_df1[lookback:].reset_index(drop= True),
                                            data_TV_train_sc_df1],
                                            axis= 1,
                                            ignore_index= True)
# train_window1 = pd.concat([data_HI_train_sc_df1.reset_index(drop= True),
#                                             data_TV_train_sc_df1],
#                                             axis= 1,
#                                             ignore_index= True)
    
train_data1= np.array(train_window1)[: -lookback]

X_train1, y_train1,= train_data1, np.array(TV_first_column1.iloc[lookback:])


X_roll1= X_train1.copy()
y_roll1= y_train1.copy()
X_train1= X_train1.reshape(X_train1.shape[0], 1, X_train1.shape[1])
X_train1, y_train1= torch.FloatTensor(X_train1), torch.FloatTensor(y_train1)


#testing data normalization
data_cycle_sc1= cycle_sc1.fit_transform(data_cycle1)
data_cycle_sc_df1= pd.DataFrame(data=data_cycle_sc1, columns=['Time'])
data_cycle_test_sc1= cycle_sc1.transform(data_cycle_test1)
data_cycle_test_sc_df1= pd.DataFrame(data=data_cycle_test_sc1, columns=['Time'])

data_HI_sc1 = HI_sc1.fit_transform(data_HI1)
data_HI_test_sc1 = HI_sc1.transform(data_HI_test1)
data_HI_test_sc_df1 = pd.DataFrame(data=data_HI_test_sc1, columns= data_HI1.columns)

data_TV_sc1 = TV_sc1.fit_transform(data_SOH1)
data_TV_test_sc1 = TV_sc1.transform(data_SOH_test1)
data_TV_test_sc_df1 = pd.DataFrame(data=data_TV_test_sc1, columns=['TV'])

for i in range(lookback):
    data_TV_test_sc_df1 ['TV'+ str(i+1)]= data_TV_test_sc_df1 ['TV'].shift(-i- 1)
    
TV_first_column2 = data_TV_test_sc_df1["TV"]
data_TV_test_sc_df1 = data_TV_test_sc_df1.drop(["TV"], axis= 1)

test_window1 = pd.concat([data_HI_test_sc_df1[lookback:].reset_index(drop= True),
                                            data_TV_test_sc_df1],
                                            axis= 1,
                                            ignore_index= True)
# train_window = pd.concat([data_HI_train_sc_df.reset_index(drop= True),
#                                             data_TV_train_sc_df],
#                                             axis= 1,
#                                             ignore_index= True)
    
test_data1= np.array(test_window1)[: -lookback]

X_test1, y_test1,= test_data1, np.array(TV_first_column2.iloc[lookback:])

global result
result = '\nEvaluation.'
result1= 'Computation Time'

start= time.time()

X_roll1= X_test1.copy()
y_roll1= y_test1.copy()
X_test1= X_test1.reshape(X_test1.shape[0], 1, X_test1.shape[1])
X_test1, y_test1= torch.FloatTensor(X_test1), torch.FloatTensor(y_test1)

def lstm_TL(input_size=21, hidden_layer_size=5, output_size=1, pre_model=pre_model1, freeze=True, verbose=True):
    model = LSTM(input_size, hidden_layer_size, output_size)
    if pre_model:
        pre_model_dict = pre_model.state_dict()
        if freeze:
            for param in pre_model_dict["linear.weight"]:
                param.requires_grad= True
    if verbose:
        print(model)
    return model

# # def lstm_TL(input_size=21, hidden_layer_size=5, output_size=1, pre_model=pre_model1, freeze=True):

# #     model = LSTM(input_size=21, hidden_layer_size=5, output_size=1)
# #     model_children = list(model.children())
# #     pre_model_children = list(pre_model.children())
    
# #     for i in range(2, len(model_children ) - 1):
# #         if type(model_children [i]) == type(pre_model_children [i]):
           
# #             model_children[i].load_state_dict(pre_model_children[i].state_dict())
            
# #             if freeze:
# #                 for param in model_children[i].parameters():
# #                     param.requires_grad = True
# #         else:
# #             print(f"Skipping layer {i} due to type mismatch.")

# #     # if verbose: print(model)    
    
# #     return model


MQLSTM_model1  = lstm_TL(input_size=21, hidden_layer_size=5, output_size=1, pre_model=pre_model1, freeze=True)

# MQLSTM_model1  = lstm(input_size=21, hidden_layer_size=5, output_size=1, pre_model=None, freeze=False, verbose=True)

optimizer = torch.optim.Adam(MQLSTM_model1.parameters(), lr = LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.1, patience=1, verbose=True)
  
# MQLSTM_model= LSTM()
loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(MQLSTM_model1.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.01, alpha=0.75, eps=1e-08,
#                                 weight_decay=0.5, momentum = 0,
#                                 centered=False, foreach=None, maximize=False, differentiable=False)
epochs = EPOCHS

train_loader1 = Data.DataLoader(dataset=Data.TensorDataset(X_train1, y_train1),
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=0,
                               )

MQLSTM_history = {'MQLSTM_train_Loss':    []}

MQLSTM_train_EPOCH_losses1 = []

for i in range(epochs):
    MQLSTM_model1.train()
    MQLSTM_train_losses1 = []
    MQLSTM_train_preds1 = []
    MQLSTM_train_targets1 = []
    for MQLSTM_train_seq1, MQLSTM_train_labels1 in train_loader1:
        optimizer.zero_grad()
        MQLSTM_train_y_pred1 = MQLSTM_model1(MQLSTM_train_seq1).squeeze(-1)
        # QLSTM_train_y_pred = np.squeeze(model(QLSTM_train_seq), -1).detach().numpy()
        MQLSTM_train_single_loss1 = loss_function(
            MQLSTM_train_y_pred1, MQLSTM_train_labels1)

        MQLSTM_train_single_loss1.backward()
        optimizer.step()

        MQLSTM_train_preds1.append(MQLSTM_train_y_pred1)
        MQLSTM_train_targets1.append(MQLSTM_train_labels1)
        MQLSTM_train_losses1.append(float(MQLSTM_train_single_loss1))
        # print("Train Step:", i, " QLSTM_train_loss: ", QLSTM_train_single_loss)
    MQLSTM_train_preds1 = [round(r)
                         for r in MQLSTM_model1.totolist1(MQLSTM_train_preds1)]
    MQLSTM_train_targets1 = MQLSTM_model1.totolist1(MQLSTM_train_targets1)
    MQLSTM_train_avg_loss1 = np.mean(MQLSTM_train_losses1)
    scheduler.step(MQLSTM_train_avg_loss1)
    
    print(
        f"Epoch {i} :  MQLSTM_train_avg_Loss1 = {MQLSTM_train_avg_loss1:2.6f} ")
    
   
        
    MQLSTM_train_EPOCH_losses1.append(MQLSTM_train_avg_loss1)
    del MQLSTM_train_preds1, MQLSTM_train_targets1
    end= time.time() 
    print(round((end-start), 2),"seconds")

MQLSTM_model1.eval()
MQLSTM_valid_y_pred1 = MQLSTM_model1(X_train1).squeeze(-1)
MQLSTM_valid_y_pred1= MQLSTM_valid_y_pred1.data.cpu().numpy().ravel()




plt.figure(num=1, figsize=(10, 6))
plt.plot(np.arange(0, len(y_train1)), y_train1.data.cpu(
).numpy().ravel(), color="darkblue", label="y actual (train)")
plt.plot(np.arange(0, len(y_train1)), MQLSTM_valid_y_pred1,
         color="darkred", label="y pred (train)")
plt.grid(True)
plt.legend(loc="best")
plt.show()

# MQLSTM_train_EPOCH_losses11 = pd.DataFrame(
#     np.array(MQLSTM_train_EPOCH_losses1).reshape(-1, 1))
# # MQLSTM_train_EPOCH_losses1 .to_csv("C:/Users/MA201_5950X/Desktop/sample codes of QLSTM2/voltage pred_21324/MQLSTM/vltg_pred_621/Training_loss3_9_533.csv")
# plt.figure(num=1, figsize=(10, 6))
# plt.plot(np.arange(0, len(MQLSTM_train_EPOCH_losses1)),
#          MQLSTM_train_EPOCH_losses1, color="red", label="training lsoss")
# plt.grid(True)
# plt.legend(loc="best")
# plt.show()

#testing
y_pred_test_T  = MQLSTM_model1(X_test1)
y_pred_test_T = y_pred_test_T.detach().numpy() 
y_pred_test_T = TV_sc1.inverse_transform(y_pred_test_T)
y_pred_test_T1 = pd.DataFrame(y_pred_test_T)
y_test_T= TV_sc1.inverse_transform(y_test1.reshape(-1, 1))
y_test_T =  pd.DataFrame(y_test_T)
# y_test_T = data_SOH1.loc[621:]

# pd.DataFrame(y_pred_test_T.reshape(-1, 1)).to_csv("C:/Users/MA201_5950X/Desktop/sample codes of QLSTM2/models_getnet/voltage_032724/2_5_611.csv")


plt.figure(num= 1, figsize= (10, 6))
plt.plot(np.arange(0, len(y_pred_test_T)), y_test_T, color= "darkblue", label= "y actual (train)")
plt.plot(np.arange (0, len(y_pred_test_T)), y_pred_test_T, color= "darkred", label= "y pred (train)")
plt.grid(True)
plt.legend(loc= "best")
plt.show()

MSE,RMSE,MAE, MAEP= loss(y_pred_test_T1,y_test_T)
# result1+= '\nRunning Time LSTM: {}'.format(end2-start2)
result += '\n\nMQLSTM_MAE: {}'.format(MAE)
result += '\nMQLSTM_RMSE: {}'.format(RMSE)
result += '\nMQLSTM_MSE: {}'.format(MSE)
result += '\nMQLSTM_MAEPE: {}'.format(MAEP)

print(result)
print(result1)