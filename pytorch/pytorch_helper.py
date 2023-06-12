import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np

from collections import OrderedDict


def loss_fn(pred, target, device):
    
    transposed_targets = target.transpose(0,1)
    k = transposed_targets[0]
    k_min = transposed_targets[1]
    k_max = transposed_targets[2]
    batch_size = target.size()[0]
    
    # 1. prob = nn.LogSoftmax(dim=1)(pred)
    # 2. for every sample in batch:
        # sum of -prob[sample][k_min, k_max] for every sample in batch
    
    prob = nn.Softmax(dim=1)(pred)
    
    total_loss = torch.tensor([0.]).to(device)
    for i in range(0, batch_size):
        indices = torch.tensor(range(k_min[i].item(), k_max[i].item()+1))
        total_loss += -torch.index_select(prob, 1, indices)[i].sum().log()
    total_loss /= batch_size
    return total_loss
   

class DeepHEC(nn.Module):
    # HE initialization is automatically applied.
    # See https://github.com/pytorch/pytorch/blob/af7dc23124a6e3e7b8af0637e3b027f3a8b3fb76/torch/nn/modules/linear.py#L101
    def __init__(self, hidden_layers, layer_size, inputs, outputs):
        super(DeepHEC, self).__init__()        
        
        layers = list()
        # input layer
        layers.append(('dense_1', nn.Linear(inputs, layer_size)))
        layers.append(('leaky_re_lu_1', nn.LeakyReLU()))
        # iteratively add more layers
        for i in range(hidden_layers - 1):
            layers.append(('dense_' + str(i+2), nn.Linear(layer_size, layer_size)))
            layers.append(('leaky_re_lu_' + str(i+2), nn.LeakyReLU()))
        # add output layer
        layers.append(('dense_' + str(hidden_layers+1), nn.Linear(layer_size, outputs)))

        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        self.sequential_layers = nn.Sequential(OrderedDict(layers))
        
    def forward(self, x):
        # nn.Sequential handles everything.
        return self.sequential_layers(x)                         



class CustomDataset(Dataset):
    def __init__(self, data, target, transform=None, device="cpu"):
        """
        Args:
            input_df (DataFrame): The input Dataframe
            label_df (DataFrame): The Label Dataframe
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = torch.from_numpy(data.values).float().to(device)
        self.target = torch.from_numpy(target.values).long().to(device)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        data = self.data.__getitem__(idx)
        target = self.target.__getitem__(idx)
        #data = torch.tensor(self.data.iloc[idx].values).float()
        #target = torch.tensor(self.target.iloc[idx].values, dtype=torch.int64)
        
        sample = (data,  target)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
def z_norm(x, train):
    train_stats = train.describe()
    train_stats = train_stats.transpose()
    mean = train_stats['mean']
    std = train_stats['std']
    return (x-mean)/std
    
def get_data_sets(path):
    return get_data_sets_by_row(path, [6,7,8])

def get_data_sets_k(path):
    return get_data_sets_by_row(path, [6])

def get_data_sets_k_minmax(path):
    return get_data_sets_by_row(path, [6,12,13])


def get_data_sets_by_row(path, rows):
    # 1. load into pd and remove unnecessary columns for training
    #df = pd.read_csv('../data/dataset.csv')
    df = pd.read_csv(path)

    # Input set simplification
    # D_PL is directly proportional to T_s. P_L and D_RS are constant
    col_names = ["D_PL(ms)","P_L(B)","D_RS(ms)"]
    df = df.drop(col_names, axis=1)

    # Get rid of cases whose channel erasure rate is too large
    df = df[df['p_e(prob)'] <= 0.2]

    cols,_ = df.shape
    #print("Number of all samples {}".format(cols))

    # Get rid of invalid configurations
    df = df[df['RI(rate)'] < float('inf')]
    df.dropna()

    # Shuffle samples
    #df = df.sample(frac=1)


    cols_valid,_ = df.shape
    #print("Number of valid samples {} out of {} ({:.2f}%)".format(cols_valid, cols, cols_valid/cols*100))

    SAMPLES = cols_valid

    # The input dataset is spit into 60% for training, 20% for validation
    # and 20% for testing.
    TRAIN_SPLIT = int(0.6 * SAMPLES)
    TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)


    # Split the data set. The second argument is an array of indices where to split 
    # the data. As two indices are provided, the data will be divided in three.
    #train, validate, test = np.split(df, [TRAIN_SPLIT, TEST_SPLIT])
    train, validate, test = np.split(df, [TRAIN_SPLIT,TEST_SPLIT])

    # Check that our splits add up correctly
    assert(len(train) + len(validate) + len(test)) == SAMPLES

    # split the dataframe into input and train/test_labels
    train_labels = train.iloc[:, rows]
    validate_labels = validate.iloc[:, rows]
    test_labels = test.iloc[:, rows]

    train = train.iloc[:,list(range(0,6))]
    validate = validate.iloc[:,list(range(0,6))]
    test = test.iloc[:,list(range(0,6))]

    normalized_train_data = z_norm(train,train)
    normalized_validate_data = z_norm(validate,train)
    normalized_test_data = z_norm(test,train)

    train_dataset = CustomDataset(normalized_train_data, train_labels)
    validate_dataset = CustomDataset(normalized_validate_data, validate_labels)
    test_dataset = CustomDataset(normalized_test_data, test_labels)
    
    
    return train_dataset, validate_dataset, test_dataset


def validate_model(model, dataloader, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validation_loss, correct_k, correct_p, correct_nc = 0, 0, 0, 0
    loss_fn_class = nn.CrossEntropyLoss()
    loss_fn_reg = nn.MSELoss()
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            #pred = model(x.to(device))
            #y = y.to(device)
            (pred_k, pred_p, pred_nc) = torch.split(model(x.to(device)), [256,256,256], dim=1)
            (y_k, y_p, y_nc) = torch.split(y, [1,1,1], dim=1)
            loss_k = loss_fn_class(pred_k, torch.flatten(y_k.to(device))).item()
            loss_p = loss_fn_class(pred_p, torch.flatten(y_p.to(device))).item()
            #loss_nc = loss_fn_reg(pred_nc, torch.flatten(y_nc.type(torch.float).to(device))).item()
            loss_nc = loss_fn_class(pred_nc, torch.flatten(y_nc.to(device))).item()
            validation_loss += loss_k + loss_p + loss_nc
            # Obtain accuracy
            correct_k += (pred_k.argmax(1) == torch.flatten(y_k.to(device))).type(torch.float).sum().item()
            correct_p += (pred_p.argmax(1) == torch.flatten(y_p.to(device))).type(torch.float).sum().item()
            correct_nc += (pred_nc.argmax(1) == torch.flatten(y_nc.to(device))).type(torch.float).sum().item()

    validation_loss /= num_batches
    correct_k /= size
    correct_p /= size
    correct_nc /= size
    return validation_loss, correct_k, correct_p, correct_nc

def validate_model_k(model, dataloader, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validation_loss, correct_k = 0., 0.
    loss_fn_class = nn.CrossEntropyLoss()
    loss_fn_reg = nn.MSELoss()
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            # Compute prediction and loss
            pred_k = model(x.to(device))
            loss = loss_fn_class(pred_k, torch.flatten(y.to(device)))
            validation_loss += loss.item()
            # Obtain accuracy
            correct_k += (pred_k.argmax(1) == torch.flatten(y.to(device))).type(torch.float).sum().item()

    validation_loss /= num_batches
    correct_k /= size
    return validation_loss, correct_k

def validate_model_k_minmax(model, dataloader, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validation_loss, correct_k = 0., 0.
    loss_fn_class = loss_fn
    loss_fn_reg = nn.MSELoss()
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            # Compute prediction and loss
            pred_k = model(x.to(device))
            loss = loss_fn_class(pred_k, y.to(device), device)
            validation_loss += loss.item()
            # Obtain accuracy (Mult for and, correct if k_min <= k <= k_max
            correct_k += ((pred_k.argmax(1) >= y.transpose(0,1)[1].to(device)).type(torch.float) * (pred_k.argmax(1) <= y.transpose(0,1)[2].to(device)).type(torch.float)).type(torch.float).sum().item()

    validation_loss /= num_batches
    correct_k /= size
    return validation_loss, correct_k



def custom_range(upper, lower): # from including 10e-upper to including 10e-lower
    r = list()
    for i in range(upper, lower):
        for j in range(10,1, -1):
            r.append(float(str(j)+"e-"+str(i)))
    r.append(float("1e-"+str(lower-1)))
    return r


def get_test_dataset(path):
    # 1. load into pd and remove unnecessary columns for training
    #df = pd.read_csv('../data/dataset.csv')
    df = pd.read_csv(path)

    # Input set simplification
    # D_PL is directly proportional to T_s. P_L and D_RS are constant
    col_names = ["D_PL(ms)","P_L(B)","D_RS(ms)"]
    df = df.drop(col_names, axis=1)

    # Get rid of cases whose channel erasure rate is too large
    df = df[df['p_e(prob)'] <= 0.2]

    cols,_ = df.shape
    #print("Number of all samples {}".format(cols))

    # Get rid of invalid configurations
    df = df[df['RI(rate)'] < float('inf')]
    df.dropna()

    # Shuffle samples
    #df = df.sample(frac=1)


    cols_valid,_ = df.shape
    #print("Number of valid samples {} out of {} ({:.2f}%)".format(cols_valid, cols, cols_valid/cols*100))

    SAMPLES = cols_valid

    # The input dataset is spit into 60% for training, 20% for validation
    # and 20% for testing.
    TRAIN_SPLIT = int(0.6 * SAMPLES)
    TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)


    # Split the data set. The second argument is an array of indices where to split
    # the data. As two indices are provided, the data will be divided in three.
    #train, validate, test = np.split(df, [TRAIN_SPLIT, TEST_SPLIT])
    train, validate, test = np.split(df, [TRAIN_SPLIT,TEST_SPLIT])

    # Check that our splits add up correctly
    assert(len(train) + len(validate) + len(test)) == SAMPLES
    
    train_labels = train.iloc[:, [6]]
    validate_labels = validate.iloc[:, [6]]
    test_labels = test.iloc[:, [6]]

    train = train.iloc[:,list(range(0,6))]
    validate = validate.iloc[:,list(range(0,6))]
    test = test.iloc[:,list(range(0,6))]

    normalized_train_data = z_norm(train,train)
    normalized_validate_data = z_norm(validate,train)
    normalized_test_data = z_norm(test,train)

    train_dataset = CustomDataset(normalized_train_data, train_labels)
    validate_dataset = CustomDataset(normalized_validate_data, validate_labels)
    test_dataset = CustomDataset(normalized_test_data, test_labels)

    return test_dataset
