# Salvatore Depasquale and Brian Dilosa | ITS365 | Final Project 

import numpy as np
import pandas as pd

import torch
import torch.onnx
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score
    )
################################################################################################

raw_data = "data.csv"

################################################################################################

columns = ([
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'attack',
    'level'
    ])

################################################################################################

data = pd.read_csv(raw_data, header = None, names = columns)

################################################################################################

def map_value(value):
    if value == 'normal':
        return 0
    else:
        return 1
    
data['attack'] = data['attack'].apply(map_value)

################################################################################################

types = data.dtypes

categorical_columns = []
categorical_dims = {}

for col in data.columns:
    if types[col] == 'object':
        l_enc = LabelEncoder()
        data[col] = l_enc.fit_transform(data[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        data.fillna(data[col].mean(), inplace = True)

data.drop(columns=['level'], inplace=True)

################################################################################################

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

################################################################################################
'''
train_loader = DataLoader(
    train_ds, 
    batch_size = 128, 
    shuffle = True, 
    num_workers = 1, 
    pin_memory = True,
    )
'''

test_loader = DataLoader(
    test_ds, 
    batch_size = 25196, 
    num_workers = 1, 
    pin_memory = True
    )

################################################################################################

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.hidden_1 = nn.Linear(41, 40)
        self.hidden_2 = nn.Linear(40, 40)
        self.hidden_3 = nn.Linear(40, 40)
        self.hidden_4 = nn.Linear(40, 20)
        self.output_layer = nn.Linear(20, 2)


    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        x = self.output_layer(x)
        return x
    
################################################################################################
'''
def train(epochs, model, lossFunction, opt):
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for data, target in train_loader:
            data, target = data.to("cuda"), target.to("cuda")
            opt.zero_grad()
            output = model(data)
            loss = lossFunction(output, target)
            loss.backward()
            opt.step()
            train_loss += loss.item()*data.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
'''

################################################################################################

'''
model = Network()
model = model.to("cuda")

lossFunction = nn.CrossEntropyLoss()

opt = optim.Adam(model.parameters(), lr = 0.01)

epochs = 10

train(epochs, model, lossFunction, opt)

torch.save(model.state_dict(), "model.pt")
'''

################################################################################################

'''
model_loaded = Network()
model_loaded.load_state_dict(torch.load("model.pt"))
model_loaded = model_loaded.to("cuda")

epochs_new = 1

train(epochs_new, model_loaded, lossFunction, opt)

torch.save(model_loaded.state_dict(), "model.pt")
'''

################################################################################################

def test(model, lossFunction):
    class_correct = list(0. for i in range(41))
    class_total = list(0. for i in range(41))
    test_loss = 0.0

    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = lossFunction(output, target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)    
        correct_tensor = pred.eq(target)
        correct = np.squeeze(correct_tensor.cpu().numpy())
        for i in range(len(data)):
            label = target[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    accuracy_0 = 100 * class_correct[0] / class_total[0]
    accuracy_1 = 100 * class_correct[1] / class_total[1]
    print('Test Accuracy of predicting that it is a normal packet: %2d%% (%2d/%2d)' % (
        accuracy_0, np.sum(class_correct[0]), np.sum(class_total[0])))
    print('Test Accuracy of predicting that is is a malicious packet: %2d%% (%2d/%2d)' % (
        accuracy_1, np.sum(class_correct[1]), np.sum(class_total[1])))
    
    confmat = confusion_matrix(target, pred)

    precision = precision_score(target, pred, average = 'weighted')

    recall = recall_score(target, pred, average = 'weighted')

    f1 = f1_score(target, pred, average = 'weighted')

    print("\nConfusion Matrix:")
    print(confmat, "\n")

    print("Precision score: ", precision, "\n")
    print("Recall score: ", recall, "\n")
    print("F1 score: ", f1, "\n")

################################################################################################

model_final = Network()
model_final.load_state_dict(torch.load("model.pt"))
model_final = model_final.to("cpu")

lossFunction = nn.CrossEntropyLoss()

test(model_final, lossFunction)

dummy_input = torch.randn(1, 41)
torch.onnx.export(model_final, dummy_input, "DNN_NSL-KDD.onnx", verbose = True)