# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:15:24 2022

@author: hosseind
"""

import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath('/Users/hosseind/Desktop/Datasets_codes/AGCRN-master')))
print(file_dir)
sys.path.append(file_dir)


import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
import random
import matplotlib
import matplotlib.pyplot as plt
from numpy import savetxt

import tensorflow as tf
from tensorflow import keras
import sklearn.preprocessing
from keras.models import Sequential
from keras.layers import BatchNormalization,Dense, Conv1D, MaxPooling1D, Flatten, GRU, LSTM
from keras.callbacks import EarlyStopping
from keras import optimizers
print(torch.__version__)
print(torch.cuda.is_available())
tf.config.list_physical_devices()
np.version.version
print(tf.__version__)


def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')
    

import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


#%% model
import torch.nn.functional as F

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])    # chon cheb_k=2 ast in loop for run nemishavad
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in  # be ebarati = "knn,bnc->bknc"
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        r, z = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, r*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = z*state + (1-z)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)    (B: batch, T: lag (khodam), N: nodes, D: input_dim = 1 (khodam))
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]  # T
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim      #(input_dim=1) man neveshtam
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T1 (lag), N, hidden (rnn_units)
        output = output[:, -1:, :, :]                      # B, 1, N, hidden (rnn_units)

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C   (B: batch, T: horizon, N: nodes, C: 1)

        return output   
    
class AGCRN_Class(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output   


#%%
'''
Always evaluate the model with MAE, RMSE, MAPE, RRSE, PNBI, and oPNBI.
Why add mask to MAE and RMSE?
    Filter the 0 that may be caused by error (such as loop sensor)
Why add mask to MAPE and MARE?
    Ignore very small values (e.g., 0.5/0.5=100%)
'''
def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def RRSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))

def CORR_torch(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(dim=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def PNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    indicator = torch.gt(pred - true, 0).float()
    return indicator.mean()

def oPNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    bias = (true+pred) / (2*true)
    return bias.mean()

def MARE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))

def SMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred)/(torch.abs(true)+torch.abs(pred)))


def MAE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def RMSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

#Root Relative Squared Error
def RRSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred-true) ** 2)), np.sqrt(np.sum((true-mean) ** 2)))

def MAPE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def PNBI_np(pred, true, mask_value=None):
    #if PNBI=0, all pred are smaller than true
    #if PNBI=1, all pred are bigger than true
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = pred-true
    indicator = np.where(bias>0, True, False)
    return indicator.mean()

def oPNBI_np(pred, true, mask_value=None):
    #if oPNBI>1, pred are bigger than true
    #if oPNBI<1, pred are smaller than true
    #however, this metric is too sentive to small values. Not good!
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = (true + pred) / (2 * true)
    return bias.mean()

def MARE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true> (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.divide(np.sum(np.absolute((true - pred))), np.sum(true))

def CORR_np(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        #B, N
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        #np.transpose include permute, B, T, N
        pred = np.expand_dims(pred.transpose(0, 2, 1), axis=1)
        true = np.expand_dims(true.transpose(0, 2, 1), axis=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(0, 1, 2, 3)
        true = true.transpose(0, 1, 2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(axis=dims)
    true_mean = true.mean(axis=dims)
    pred_std = pred.std(axis=dims)
    true_std = true.std(axis=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(axis=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation

def All_Metrics(pred, true, mask1, mask2):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = MAE_np(pred, true, mask1)
        rmse = RMSE_np(pred, true, mask1)
        mape = MAPE_np(pred, true, mask2)
        rrse = RRSE_np(pred, true, mask1)
        corr = 0
        #corr = CORR_np(pred, true, mask1)
        #pnbi = PNBI_np(pred, true, mask1)
        #opnbi = oPNBI_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae  = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        mape = MAPE_torch(pred, true, mask2)
        rrse = RRSE_torch(pred, true, mask1)
        corr = CORR_torch(pred, true, mask1)
        #pnbi = PNBI_torch(pred, true, mask1)
        #opnbi = oPNBI_torch(pred, true, mask2)
    else:
        raise TypeError
    return mae, rmse, mape, rrse, corr

def MAE_np_m(pred, true, mask_value=None):
    if mask_value[1,1] != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def RMSE_np_m(pred, true, mask_value=None):
    if mask_value[1,1] != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

def MAPE_np_m(pred, true, mask_value=None):
    if mask_value[1,1] != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

#Root Relative Squared Error
def RRSE_np_m(pred, true, mask_value=None):
    if mask_value[1,1] != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred-true) ** 2)), np.sqrt(np.sum((true-mean) ** 2)))

def All_Metrics_m(pred, true, mask1, mask2):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = MAE_np_m(pred, true, mask1)
        rmse = RMSE_np_m(pred, true, mask1)
        mape = MAPE_np_m(pred, true, mask2)
        rrse = RRSE_np_m(pred, true, mask1)
        corr = 0
        #corr = CORR_np(pred, true, mask1)
        #pnbi = PNBI_np(pred, true, mask1)
        #opnbi = oPNBI_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae  = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        mape = MAPE_torch(pred, true, mask2)
        rrse = RRSE_torch(pred, true, mask1)
        corr = CORR_torch(pred, true, mask1)
        #pnbi = PNBI_torch(pred, true, mask1)
        #opnbi = oPNBI_torch(pred, true, mask2)
    else:
        raise TypeError
    return mae, rmse, mape, rrse, corr

def SIGIR_Metrics(pred, true, mask1, mask2):
    rrse = RRSE_torch(pred, true, mask1)
    corr = CORR_torch(pred, true, 0)
    return rrse, corr
'''
if __name__ == '__main__':
    pred = torch.Tensor([1, 2, 3,4])
    true = torch.Tensor([2, 1, 4,5])
    print(All_Metrics(pred, true, None, None))
'''

#%%Trainer
import math
import time
import copy
import logging

def get_logger(root, name=None, debug=True):
    #when debug is true, show DEBUG and INFO in screen
    #when debug is false, show DEBUG in file and info in both screen&file
    #INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    #critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        logfile = root + '/run.log'
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger


'''if __name__ == '__main__':
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    print(time)
    logger = get_logger('./log.txt', debug=True)
    logger.debug('this is a {} debug message'.format(1))
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
'''

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = self.args.log_dir + '/best_model.pth'
        self.loss_figure_path = self.args.log_dir + '/loss.png'
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.model(data, target, teacher_forcing_ratio=0.)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output, label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            loss = self.loss(output, label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            if (not os.path.exists(args.log_dir)):
                os.makedirs(args.log_dir)
            torch.save(best_model, os.path.join(self.best_path))
            self.logger.info("Saving current best model to " + self.best_path)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        y_true, y_pred = self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)
        return y_true, y_pred

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output = model(data, target, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        np.save(args.log_dir + '/{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save(args.log_dir + '/{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))
        return y_true, y_pred

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
    
#%% dataloader
import torch.utils.data
import pandas as pd

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('/Users/hosseind/Desktop/Datasets_codes/AGCRN-master/data/PEMS04/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('/Users/hosseind/Desktop/Datasets_codes/AGCRN-master/data/PEMS08/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data = pd.read_csv(r'/Users/hosseind/Desktop/Datasets_codes/PeMSD7/PeMSD7_speed.csv')
        data = data.values
        #data = data[0::10,:]
    elif dataset == 'WADI':
        data = pd.read_csv(r'/Users/hosseind/Desktop/Datasets_codes/WADI/WADI.A2_19 Nov 2019/WADI_14days_new.csv')
        data = data.values
        data = data[0::10,:]    
    elif dataset == 'WaterTank':
        data = pd.read_csv(r'/Users/hosseind/Desktop/Datasets_codes/GraphDataset-master/measurements_1.csv')
        data2 = pd.read_csv(r'/Users/hosseind/Desktop/Datasets_codes/GraphDataset-master/measurements_2.csv')
        data3 = pd.read_csv(r'/Users/hosseind/Desktop/Datasets_codes/GraphDataset-master/measurements_3.csv')
        data = data.values
        data2 = data2.values
        data3 = data3.values
        data = np.concatenate((data, data2, data3), axis=0)
        data = data[:,1:101]    
    elif dataset == 'SWAT':
        data1 = pd.read_excel(r'/Users/hosseind/Desktop/Datasets_codes/WADI_SWAT/SWaT.A1 & A2_Dec 2015/SWaT_Dataset_Normal_v0.xlsx')
        data2 = pd.read_excel(r'/Users/hosseind/Desktop/Datasets_codes/WADI_SWAT/SWaT.A1 & A2_Dec 2015/SWaT_Dataset_Attack_v0.xlsx')
        data1 = data1.values
        data1 = data1[0::12,:]
        data1 = data1[:,1:]
        data2 = data2.values
        data2 = data2[0::12,:]
        data2 = data2[:,1:]
        data3 = np.concatenate((data1, data2[1:,:]), axis=0)
        data3 = data3[1:,:]
        data3 = np.where(data3=='Normal', 0, data3) 
        data3 = np.where(data3=='Attack', 1, data3) 
        data3 = np.where(data3=='A ttack', 1, data3) 
        data = data3.astype('float32')
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

class NScaler(object):
    def transform(self, data):
        return data
    def inverse_transform(self, data):
        return data

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

class MinMax01Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min

class ColumnMinMaxScaler():
    #Note: to use this scale, must init the min and max with column min and column max
    def __init__(self, min, max):
        self.min = min
        self.min_max = max - self.min
        self.min_max[self.min_max==0] = 1
    def transform(self, data):
        print(data.shape, self.min_max.shape)
        return (data - self.min) / self.min_max

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min_max = torch.from_numpy(self.min_max).to(data.device).type(torch.float32)
            self.min = torch.from_numpy(self.min).to(data.device).type(torch.float32)
        #print(data.dtype, self.min_max.dtype, self.min.dtype)
        return (data * self.min_max + self.min)

def one_hot_by_column(data):
    #data is a 2D numpy array
    len = data.shape[0]
    for i in range(data.shape[1]):
        column = data[:, i]
        max = column.max()
        min = column.min()
        #print(len, max, min)
        zero_matrix = np.zeros((len, max-min+1))
        zero_matrix[np.arange(len), column-min] = 1
        if i == 0:
            encoded = zero_matrix
        else:
            encoded = np.hstack((encoded, zero_matrix))
    return encoded


def minmax_by_column(data):
    # data is a 2D numpy array
    for i in range(data.shape[1]):
        column = data[:, i]
        max = column.max()
        min = column.min()
        column = (column - min) / (max - min)
        column = column[:, np.newaxis]
        if i == 0:
            _normalized = column
        else:
            _normalized = np.hstack((_normalized, column))
    return _normalized


'''if __name__ == '__main__':

    test_data = np.array([[0,0,0, 1], [0, 1, 3, 2], [0, 2, 1, 3]])
    print(test_data)
    minimum = test_data.min(axis=1)
    print(minimum, minimum.shape, test_data.shape)
    maximum = test_data.max(axis=1)
    print(maximum)
    print(test_data-minimum)
    test_data = (test_data-minimum) / (maximum-minimum)
    print(test_data)
    print(0 == 0)
    print(0.00 == 0)
    print(0 == 0.00)
    #print(one_hot_by_column(test_data))
    #print(minmax_by_column(test_data))


if __name__ == '__main__':
    from data.load_raw_data import Load_Sydney_Demand_Data
    path = '../data/1h_data_new3.csv'
    data = Load_Sydney_Demand_Data(path)
    print(data.shape)
    X, Y = Add_Window_Horizon(data, horizon=2)
    print(X.shape, Y.shape)
'''
def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    
    ## for Week End and Week Day divide of PEMSD8
    #WD, WE = np.zeros((1,170)),np.zeros((1,170))
    #WD = np.append(WD, data[288*(0):288*1,:,0], axis=0)
    #for i in range(9):
    #    j = i+1
    #    k = (j*7)-4
    #    z = (i*7)+1
    #    WD = np.append(WD, data[288*(k):288*(k+5),:,0], axis=0)
    #    WE = np.append(WE, data[288*(z):288*(z+2),:,0],axis=0)
    #WD = np.delete(WD, 0, 0)
    #WE = np.delete(WE, 0, 0)
    #WD = np.reshape(WD, (-1,170,1))
    #WE = np.reshape(WE, (-1,170,1))
    ##spilit dataset by days or by ratio
    #data_train, data_val, data_test = split_data_by_ratio(WD, 0.25, 0.0001)
    #data_test = WE
    
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler

'''
if __name__ == '__main__':
    import argparse
    #MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    DATASET = 'SIGIR_electric'
    if DATASET == 'MetrLA':
        NODE_NUM = 207
    elif DATASET == 'BikeNYC':
        NODE_NUM = 128
    elif DATASET == 'SIGIR_solar':
        NODE_NUM = 137
    elif DATASET == 'SIGIR_electric':
        NODE_NUM = 321
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True)
'''
#%%
#*************************************************************************#
Mode = 'Train'
DEBUG = 'True'
DATASET = 'PEMSD8'      #PEMSD4 or PEMSD8 or PEMSD7 or WaterTank
DEVICE = 'cuda:0'
MODEL = 'AGCRN'

#get configuration
config_file = '/Users/hosseind/Desktop/Datasets_codes/AGCRN-master/model/{}_{}.conf'.format(DATASET, MODEL)
print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
#args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args = args.parse_args()
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

if DATASET == 'WaterTank':
    args.num_nodes = 100 ## PEMSD7 has 228 nodes, PEMSD8 has 170 nodes, PEMSD4 has 307 nodes, WaterTank has 100 nodes
    args.dataset = 'WaterTank'
    
if DATASET == 'PEMSD7':
    args.num_nodes = 228 ## PEMSD7 has 228 nodes, PEMSD8 has 170 nodes, PEMSD4 has 307 nodes, WaterTank has 100 nodes
    args.dataset = 'PEMSD7'
    
if DATASET == 'SWAT':
    args.num_nodes = 51 ## PEMSD7 has 228 nodes, PEMSD8 has 170 nodes, PEMSD4 has 307 nodes, WaterTank has 100 nodes, SWAT has 51 nodes
    args.dataset = 'SWAT'
    args.val_ratio = 0.2
    args.test_ratio = 0.48

args.rnn_units = 64
args.horizon = 1
args.early_stop_patience = 10
args.epochs = 100
args.debug = False # False to save best model
args.normalizer = 'max01'
args.lag = 12
args.embed_dim = 2
args.cheb_k = 1


#init model
model = AGCRN(args)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#load dataset
#train_loader, val_loader, test_loader, scaler = get_dataloader(args,
#                                                               normalizer=args.normalizer,
#                                                               tod=args.tod, dow=False, weather=False, single=False)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
#current_dir = os.path.dirname(os.path.realpath('//home.ansatt.ntnu.no/hosseind/Desktop/AGCRN-master'))
#log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = '/Users/hosseind/Desktop/Datasets_codes/AGCRN-master/experiment/' + args.dataset

#start training
trainer = Trainer(model, loss, optimizer, train_dataloader_f, val_dataloader_f, test_dataloader_f, scaler,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'Train':
    y_true, y_pred = trainer.train()
elif args.mode == 'Test':
    model.load_state_dict(torch.load('/Users/hosseind/Desktop/Datasets_codes/AGCRN-master/experiment/' + args.dataset + '/' + 'best_model.pth'))
    print("Load saved model")
    y_true, y_pred = trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError

i = 10
plt.plot(range(len(y_true[:,0,i,0])),y_true[:,0,i,0],range(len(y_pred[:,0,i,0])),y_pred[:,0,i,0])


# prints the parameters in the model caculate the all parameters number
i=0
for parameter in model.parameters():
    print(parameter.size())
    i=i+1
    if i==1:
        A = parameter
    #print(parameter)
    
for name, param in model.named_parameters():
    if param.requires_grad:
        #print(name, param.data)
        print(name)
    
A = A.data.numpy()
A = np.multiply(A,np.transpose(A))
np.savetxt(r'/Users/hosseind/Desktop/Datasets_codes/GraphDataset-master/connection_python.csv', A, delimiter=',')

import scipy.io
scipy.io.savemat(r'/Users/hosseind/Desktop/Datasets_codes/GraphDataset-master/connection_python2.mat', {"data": A })


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
count_parameters(model)

'''
# Creating Faulty signal
BiasF_tr , Label_tr = BiasFaultPM(15,440,10,train_size)
(len(Label_tr)-np.count_nonzero(Label_tr[:,0]))/len(Label_tr)
BiasF_tst , Label_tst = BiasFault(3,15,4,test_size)
(len(Label_tst)-np.count_nonzero(Label_tst[:,0]))/len(Label_tst)
'''
#%%

normalizer=args.normalizer
tod=args.tod
dow=False
weather=False
single=False

def get_dataloader_f(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    
    #for SWAT
    Label = data[:,51]
    data = data[:,:51]
    
    
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    
    
    # create repeatitive sets #1
    data_train = np.append(data_train, data_train, axis=0)
    data_val = np.append(data_val, data_val, axis=0)
    data_test = np.append(data_test, data_test, axis=0)
    
    
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    

    #for SWAT
    train_dataloader_f = data_loader(x_tra, y_tra, args.batch_size, shuffle=False, drop_last=False)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader_f = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=False)
    test_dataloader_f = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    


    # Creating Faulty signal 
    BiasF_tr , Label_tr = DriftFaultPM(3,35,1,data_train.shape[1],8,2,data_train.shape[0])
    (len(Label_tr)-np.count_nonzero(Label_tr[:,0]))/len(Label_tr)
    BiasF_val , Label_val = DriftFaultPM(3,1,1,data_val.shape[1],8,2,data_val.shape[0])
    (len(Label_val)-np.count_nonzero(Label_val[:,0]))/len(Label_val)
    BiasF_tst , Label_tst = DriftFaultPM(3,1,0,data_test.shape[1],8,2,data_test.shape[0])
    (len(Label_tst)-np.count_nonzero(Label_tst[:,0]))/len(Label_tst)

    # Creating Faulty signal 
    BiasF_tr , Label_tr = BiasFaultPM(3,35,1,data_train.shape[1],8,2,data_train.shape[0])
    (len(Label_tr)-np.count_nonzero(Label_tr[:,0]))/len(Label_tr)
    BiasF_val , Label_val = BiasFaultPM(3,1,1,data_val.shape[1],8,2,data_val.shape[0])
    (len(Label_val)-np.count_nonzero(Label_val[:,0]))/len(Label_val)
    BiasF_tst , Label_tst = BiasFaultPM(3,1,0,data_test.shape[1],8,2,data_test.shape[0])
    (len(Label_tst)-np.count_nonzero(Label_tst[:,0]))/len(Label_tst)

    # Noise Fault
    BiasF_tr , Label_tr = NoiseFaultPM(3,35,1,data_train.shape[1],8,2,data_train.shape[0])
    (len(Label_tr)-np.count_nonzero(Label_tr[:,0]))/len(Label_tr)
    BiasF_val , Label_val = NoiseFaultPM(3,1,1,data_train.shape[1],8,2,data_val.shape[0])
    (len(Label_val)-np.count_nonzero(Label_val[:,0]))/len(Label_val)
    BiasF_tst , Label_tst = NoiseFaultPM(3,1,1,data_train.shape[1],8,2,data_test.shape[0])
    (len(Label_tst)-np.count_nonzero(Label_tst[:,0]))/len(Label_tst)


    # Creating Faulty signal 
    BiasF_tr , Label_tr = FreezeFaultPM(3,35,1,data_train.shape[1],200,300,data_train.shape[0])
    (len(Label_tr)-np.count_nonzero(Label_tr[:,0]))/len(Label_tr)
    BiasF_val , Label_val = FreezeFaultPM(3,1,1,data_train.shape[1],200,300,data_val.shape[0])
    (len(Label_val)-np.count_nonzero(Label_val[:,0]))/len(Label_val)
    BiasF_tst , Label_tst = FreezeFaultPM(3,1,1,data_train.shape[1],200,300,data_test.shape[0])
    (len(Label_tst)-np.count_nonzero(Label_tst[:,0]))/len(Label_tst)
    # Freeze
    a = np.where(BiasF_tr == 1)
    traina = data_train[:,:,0]
    count = 0
    for i in a[0]:
        BiasF_tr[i-1,a[1][count]] = traina[i-1,a[1][count]]
        BiasF_tr[i,a[1][count]] = traina[i,a[1][count]]
        count = count +1
    for j in range(data_train.shape[1]):
        for i in range(len(traina)):
            if BiasF_tr[i-1,j] == 0:
                if BiasF_tr[i,j] != 0:
                    holder = BiasF_tr[i,j]
                    ind_h = i
            else:
                if BiasF_tr[i,j] == 0:
                    ind_e = i
                    BiasF_tr[ind_h:ind_e,j] = BiasF_tr[ind_h:ind_e,j] - holder
    BiasF_tr = -BiasF_tr 


    b = np.where(BiasF_tst == 1)
    testa = data_test[:,:,0]
    count = 0
    for i in b[0]:
        BiasF_tst[i-1,b[1][count]] = testa[i-1,b[1][count]]
        BiasF_tst[i,b[1][count]] = testa[i,b[1][count]]
        count = count +1
    for j in range(data_train.shape[1]):
        for i in range(len(testa)):
            if BiasF_tst[i-1,j] == 0:
                if BiasF_tst[i,j] != 0:
                    holder = BiasF_tst[i,j]
                    ind_h = i
            else:
                if BiasF_tst[i,j] == 0:
                    ind_e = i
                    BiasF_tst[ind_h:ind_e,j] = BiasF_tst[ind_h:ind_e,j] - holder
    BiasF_tst = -BiasF_tst



    # continue
    x_tra_f, y_tra_f = Add_Window_Horizon(data_train+np.expand_dims(BiasF_tr, axis=2), args.lag, args.horizon, single)
    _, Label_tr_f = Add_Window_Horizon(Label_tr, args.lag, args.horizon, single)
    x_val_f, y_val_f = Add_Window_Horizon(data_val+np.expand_dims(BiasF_val, axis=2), args.lag, args.horizon, single)
    _, Label_val_f = Add_Window_Horizon(Label_val, args.lag, args.horizon, single)
    x_test_f, y_test_f = Add_Window_Horizon(data_test+np.expand_dims(BiasF_tst, axis=2), args.lag, args.horizon, single)
    _, Label_tst_f = Add_Window_Horizon(Label_tst, args.lag, args.horizon, single)
        

    
    '''
    
    ##############get dataloader######################
    train_dataloader_f = data_loader(x_tra_f, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader_f = data_loader(x_val_f, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader_f = data_loader(x_test_f, y_test, args.batch_size, shuffle=False, drop_last=False)
    
    #start training
    trainer = Trainer(model, loss, optimizer, train_dataloader_f, val_dataloader_f, test_dataloader_f, scaler,
                      args, lr_scheduler=lr_scheduler)
    if args.mode == 'Train':
        y_true, y_pred = trainer.train()
    
    '''
    
    
    
    ##############get dataloader######################
    train_dataloader_f = data_loader(x_tra_f, y_tra, args.batch_size, shuffle=False, drop_last=False)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader_f = data_loader(x_val_f, y_val, args.batch_size, shuffle=False, drop_last=False)
    test_dataloader_f = data_loader(x_test_f, y_test, args.batch_size, shuffle=False, drop_last=False)
    
    _, y_pred_tr = trainer.test(model, trainer.args, train_dataloader_f, scaler, trainer.logger)
    _, y_pred_val = trainer.test(model, trainer.args, val_dataloader_f, scaler, trainer.logger)
    tic()
    _, y_pred_tst = trainer.test(model, trainer.args, test_dataloader_f, scaler, trainer.logger)
    toc()
    
    #i = 10
    #plt.plot(range(len(y_true[:,0,i,0])),y_true[:,0,i,0],range(len(y_pred_tst[:,0,i,0])),y_pred_tst[:,0,i,0])
    #plt.plot(range(len(y_true[:,0,i,0])),scaler.inverse_transform(BiasF_tst)[12:,i])
    #plt.plot(range(len(y_true[:,0,i,0])),scaler.inverse_transform(y_test_f)[:,0,i,0])
    #scaler.inverse_transform(BiasF_tst)
    
    #savetxt(r'/Users/hosseind/Desktop/Datasets_codes/data/y_test_f.csv', scaler.inverse_transform(y_test_f)[:,0,:,0], delimiter=',')
    #savetxt(r'/Users/hosseind/Desktop/Datasets_codes/data/y_true.csv', y_true[:,0,:,0], delimiter=',')
    #savetxt(r'/Users/hosseind/Desktop/Datasets_codes/data/y_pred_tst.csv', y_pred_tst[:,0,:,0], delimiter=',')
    #savetxt(r'/Users/hosseind/Desktop/Datasets_codes/data/BiasF_tst.csv', BiasF_tst, delimiter=',')
    #savetxt(r'/Users/hosseind/Desktop/Datasets_codes/data/Label_tst.csv', Label_tst, delimiter=',')

    #y_pred = torch.cat((y_pred_tr, y_pred_val), 0)
    #y_pred = torch.cat((y_pred, y_pred_tst), 0)
    
    #y_pred = y_pred.numpy()
    
    #y_fault = np.append(y_tra_f, y_val_f, axis=0)
    #y_fault = np.append(y_fault, y_test_f, axis=0)
    
    
    args.lag = 6 # for classifier
       
    res_tr = abs(np.subtract(scaler.transform(y_pred_tr), y_tra_f)) # [20:,:,:,:], 20 refers to lag difference
    res_val = abs(np.subtract(scaler.transform(y_pred_val), y_val_f))
    res_tst = abs(np.subtract(scaler.transform(y_pred_tst), y_test_f))
    
    
    
    
    #for SWAT
    args.lag = 6 # for classifier
       
    res_tr = abs(np.subtract(scaler.transform(y_pred_tr), y_tra)) # [20:,:,:,:], 20 refers to lag difference
    res_val = abs(np.subtract(scaler.transform(y_pred_val), y_val))
    res_tst = abs(np.subtract(scaler.transform(y_pred_tst), y_test))
    
    res_tr = res_tr.numpy()
    res_val = res_val.numpy()
    res_tst = res_tst.numpy()
    
    res_tr_f, _ = Add_Window_Horizon(res_tr[:,0,:,:], args.lag, args.horizon, single)
    res_val_f, _ = Add_Window_Horizon(res_val[:,0,:,:], args.lag, args.horizon, single)
    res_tst_f, _ = Add_Window_Horizon(res_tst[:,0,:,:], args.lag, args.horizon, single)
    
    
    
    
    #Normalization
    std_scale = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(np.concatenate((res_tr, res_val), axis=0)[:,0,:,0])
    res_tr = std_scale.transform(res_tr[:,0,:,0])
    res_tr = np.expand_dims(res_tr, axis=1)
    res_tr = np.expand_dims(res_tr, axis=3)
    res_val = std_scale.transform(res_val[:,0,:,0])
    res_val = np.expand_dims(res_val, axis=1)
    res_val = np.expand_dims(res_val, axis=3)
    res_tst = std_scale.transform(res_tst[:,0,:,0])
    res_tst = np.expand_dims(res_tst, axis=1)
    res_tst = np.expand_dims(res_tst, axis=3)
    
    res_tr_f, _ = Add_Window_Horizon(res_tr[:,0,:,:], args.lag, args.horizon, single)
    res_val_f, _ = Add_Window_Horizon(res_val[:,0,:,:], args.lag, args.horizon, single)
    res_tst_f, _ = Add_Window_Horizon(res_tst[:,0,:,:], args.lag, args.horizon, single)
    
    '''
    res_tr_f, _ = Add_Window_Horizon(res_tr[:,0,:,:].numpy(), args.lag, args.horizon, single)
    res_val_f, _ = Add_Window_Horizon(res_val[:,0,:,:].numpy(), args.lag, args.horizon, single)
    res_tst_f, _ = Add_Window_Horizon(res_tst[:,0,:,:].numpy(), args.lag, args.horizon, single)
    '''
       
    #virtual sensor performance
    i = 29
    plt.plot(range(len(y_pred_tst[:,0,i,0])),scaler.transform(y_pred_tst[:,0,i,0]), range(len(y_test_f[:,0,i,0])),y_test_f[:,0,i,0], range(len(Label_tst_f[:,0,i+1])),Label_tst_f[:,0,i+1],   range(len(data_test[args.lag:,i,0])),data_test[args.lag:,i,0])
    
    args.lag = 12
    Mask = Label_tst_f[:,0,1:]*10000 # Only healthy readings
    Mask = (abs(1-Label_tst_f[:,0,1:]))*10000 # Only Faults
    mae, rmse, mape, _, _ = All_Metrics_m(y_pred_tst[:, 0,:,0].numpy(), scaler.inverse_transform(data_test[args.lag:,:,0]), Mask, Mask)
    rmse  = RMSE_np_m(y_pred_tst[:, 0,:,0].numpy(), scaler.inverse_transform(data_test[args.lag:,:,0]), Mask)
    
    plt.plot(y_pred_tst[:, 0,19,0].numpy()-scaler.inverse_transform(data_test[args.lag:,19,0]))
    
    
    i = 19
    plt.plot(range(len(res_tst[:,0,i,0])),res_tst[:,0,i,0], range(len(Label_tst_f[:,0,i+1])),Label_tst_f[:,0,i+1])
    plt.plot(range(len(y_pred_tst[:,0,i,0])),scaler.transform(y_pred_tst[:,0,i,0]), range(len(y_test_f[:,0,i,0])),y_test_f[:,0,i,0], range(len(Label_tst_f[:,0,i+1])),Label_tst_f[:,0,i+1])
    
    
    ##############get dataloader######################
    train_dataloader_ff = data_loader(res_tr_f, np.expand_dims(Label_tr_f[args.lag-1:-1,:,1:], axis=3), args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader_ff = data_loader(res_val_f, np.expand_dims(Label_val_f[args.lag-1:-1,:,1:], axis=3), args.batch_size, shuffle=False, drop_last=True)
    test_dataloader_ff = data_loader(res_tst_f, np.expand_dims(Label_tst_f[args.lag-1:-1,:,1:], axis=3), args.batch_size, shuffle=False, drop_last=False)
    
    return train_dataloader_ff, val_dataloader_ff, test_dataloader_ff, scaler


#for SWAT
M = args.num_nodes*2

# RNN model. Early stopping mechanisim considered to avoid overfitting (20% of train set considered as validation set)
def RNN(X,Y):
    model = Sequential()
    # Input layer
    #model.add(GRU(M, return_sequences = True, input_shape = [X.shape[1], X.shape[2]]))
    # Hidden layer
    #model.add(GRU(M)) 
    model.add(Dense(M, input_dim=args.num_nodes*args.lag, activation='tanh'))
    model.add(Dense(M, activation='tanh'))
    model.add(Dense(1, activation='sigmoid')) 
    optim = keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=optim)
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, 
                       verbose=1, patience=15, restore_best_weights=True)
    # fit model
    hist = model.fit(X,Y, epochs=400, batch_size=32, verbose=1, 
           validation_split=0.2, shuffle=True, callbacks=[es])
    return hist

a = 5000
Label_f = Label[len(data_train)+len(data_val):,:]
_, Label_tst_f = Add_Window_Horizon(Label_f[a:,:], 12, args.horizon, single)
_, Label_f = Add_Window_Horizon(Label_f, 12, args.horizon, single)
Label_f, _ = Add_Window_Horizon(Label_f, 6, args.horizon, single)
Label_f = Label_f[:,5,0,0]

# RNN
res_tst_f = np.reshape(res_tst_f, (res_tst_f.shape[0], res_tst_f.shape[1], -1))

plt.plot(res_tst_f[:1000,0,1])
plt.plot(y_test[:1000,0,1,0])
plt.plot(scaler.transform(y_pred_tst)[:1000,0,1,0])

# fit RNN model
with tf.device('/CPU:0'):
    history = RNN(res_tst_f.reshape(-1,args.lag*args.num_nodes)[:a,:], Label_f[:a])

Predict_rnn = history.model.predict(np.flip(res_tst_f,1).reshape(-1,args.lag*args.num_nodes)[a:,:])

plt.plot(Label_tst_f[6-1:,0,0])
plt.plot(Predict_rnn[:,0])




M = args.num_nodes*2

# RNN model. Early stopping mechanisim considered to avoid overfitting (20% of train set considered as validation set)
def RNN(X,Y):
    model = Sequential()
    # Input layer
    #model.add(GRU(M, return_sequences = True, input_shape = [X.shape[1], X.shape[2]]))
    # Hidden layer
    #model.add(GRU(M)) 
    model.add(Dense(M, input_dim=args.num_nodes*args.lag, activation='tanh'))
    model.add(Dense(M, activation='tanh'))
    model.add(Dense(args.num_nodes, activation='sigmoid')) 
    optim = keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=optim)
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, 
                       verbose=1, patience=5, restore_best_weights=True)
    # fit model
    hist = model.fit(X,Y, epochs=400, batch_size=32, verbose=1, 
           validation_split=0.2, shuffle=True, callbacks=[es])
    return hist

res_f = np.append(res_tr_f, res_val_f, axis=0)
Label_f = np.append(Label_tr_f[args.lag-1:-1,:,:], Label_val_f[args.lag-1:-1,:,:], axis=0)

# RNN
res_f = np.reshape(res_f, (res_f.shape[0], res_f.shape[1], -1)) # (samples, window_size, number_of_features)
res_tst_f = np.reshape(res_tst_f, (res_tst_f.shape[0], res_tst_f.shape[1], -1))

a = np.flip(res_f, 1)

# create repeatitive sets #2
a = np.append(a, a, axis=0)
Label_f = np.append(Label_f, Label_f, axis=0)


tic()
# fit RNN model
with tf.device('/CPU:0'):
    history = RNN(a.reshape(-1,args.lag*args.num_nodes), Label_f[:,0,1:])
toc()

trainScore_rnn = history.model.evaluate(a.reshape(-1,12*args.num_nodes), Label_f[:,0,1:], verbose=0)
print('Train Score: %.5f MSE' % (trainScore_rnn))
testScore_rnn = history.model.evaluate(np.flip(res_tst_f, 1).reshape(-1,args.lag*args.num_nodes), Label_tst_f[args.lag-1:-1,0,1:], verbose=0)
print('Test Score: %.5f MSE' % (testScore_rnn))

tic()
Predict_rnn = history.model.predict(np.flip(res_tst_f,1).reshape(-1,args.lag*args.num_nodes))
toc()
#%%
M = args.num_nodes

# RNN model. Early stopping mechanisim considered to avoid overfitting (20% of train set considered as validation set)
def RNN(X,Y):
    model = Sequential()
    # Input layer
    model.add(GRU(M, return_sequences = True, input_shape = [X.shape[1], X.shape[2]]))
    # Hidden layer
    model.add(GRU(M)) 
    model.add(Dense(args.num_nodes, activation='sigmoid')) 
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optim)
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, 
                       verbose=1, patience=10, restore_best_weights=True)
    # fit model
    hist = model.fit(X,Y, epochs=200, batch_size=64, verbose=1, 
           validation_split=0.2, shuffle=False, callbacks=[es])
    return hist

# CNN model. Early stopping mechanisim considered to avoid overfitting (20% of train set considered as validation set)
def RNN(X,Y):
    model = Sequential()
    model.add(Conv1D(filters=args.num_nodes, kernel_size=3, activation='tanh'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(args.num_nodes, activation='sigmoid')) 
    optim = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='binary_crossentropy', optimizer=optim)
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, 
                       verbose=1, patience=5, restore_best_weights=True)
    # fit model
    hist = model.fit(X,Y, epochs=200, batch_size=512, verbose=1, 
           validation_split=0.2, shuffle=True, callbacks=[es])
    return hist

# CNN&RNN model. Early stopping mechanisim considered to avoid overfitting (20% of train set considered as validation set)
def RNN(X,Y):
    model = Sequential()
    model.add(Conv1D(filters=args.num_nodes, kernel_size=3, padding='same', activation='tanh'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(args.num_nodes))
    model.add(Dense(args.num_nodes, activation='sigmoid')) 
    optim = tf.keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optim)
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, 
                       verbose=1, patience=5, restore_best_weights=True)
    # fit model
    hist = model.fit(X,Y, epochs=200, batch_size=64, verbose=1, 
           validation_split=0.2, shuffle=True, callbacks=[es])
    return hist


res_f = np.append(res_tr_f, res_val_f, axis=0)
Label_f = np.append(Label_tr_f[args.lag-1:-1,:,:], Label_val_f[args.lag-1:-1,:,:], axis=0)

# RNN
res_f = np.reshape(res_f, (res_f.shape[0], res_f.shape[1], -1)) # (samples, window_size, number_of_features)
res_tst_f = np.reshape(res_tst_f, (res_tst_f.shape[0], res_tst_f.shape[1], -1))

a = np.flip(res_f, 1)

# create repeatitive sets #2
a = np.append(a, a, axis=0)
Label_f = np.append(Label_f, Label_f, axis=0)


# fit RNN model
with tf.device('/CPU:0'):
    history = RNN(a, Label_f[:,0,1:])

trainScore_rnn = history.model.evaluate(a, Label_f[:,0,1:], verbose=0)
print('Train Score: %.5f MSE' % (trainScore_rnn))
testScore_rnn = history.model.evaluate(np.flip(res_tst_f, 1), Label_tst_f[args.lag-1:-1,0,1:], verbose=0)
print('Test Score: %.5f MSE' % (testScore_rnn))

Predict_rnn = history.model.predict(np.flip(res_tst_f,1))

i=63
plt.plot(range(len(Label_tst_f[args.lag-1:-1,0,i+1])),Label_tst_f[args.lag-1:-1,0,i+1],label="Label")
plt.plot(range(len(Predict_rnn[:,i])),Predict_rnn[:,i], label="Predict_RNN")
plt.legend()
plt.show()


PD = []
PF = []
eps, ep10 = -0.01, 0
while eps < 1.2:
    eps = eps + 0.001
    if eps > ep10:
        print(eps)
        ep10 = ep10 + 0.1
    mse01 = np.where(Predict_rnn<eps, 0, 1) 
    #False alarm and detection
    Pf, Pd = 0, 0
    for k in range(len(Predict_rnn)):
        if (Label_tst_f[k+args.lag-1,0,0] > 0.5):
            if sum(mse01[k,:])>0.5:
                Pf = Pf + 1
        else:
            if sum(mse01[k,:])>0.5:
                Pd = Pd + 1                
    Pf, Pd = Pf/sum(Label_tst_f[args.lag-1:-1,0,0]), Pd/(len(Predict_rnn)-sum(Label_tst_f[args.lag-1:-1,0,0]))
    PF.append(Pf), PD.append(Pd)
PD = np.array(PD)
PF = np.array(PF)



plt.plot(PF,PD,label="Probability of Detection")
plt.xscale('log',base=10) 
plt.legend()
plt.show()

plt.savefig('untitled12_mlp.png')



# Also including probability of identification
PD = []
PF = []
PI = []
eps, ep10 = -0.01, 0
while eps < 1.5:
    eps = eps + 0.001
    if eps > ep10:
        print(eps)
        ep10 = ep10 + 0.1
    mse01 = np.where(Predict_rnn<eps, 0, 1) 
    #False alarm and detection
    Pf, Pd, Pi = 0, 0, 0
    for k in range(len(Predict_rnn)):
        if (Label_tst_f[k+args.lag-1,0,0] > 0.5):
            if sum(mse01[k,:])>0.5:
                Pf = Pf + 1
        else:
            if sum(mse01[k,:])>0.5:
                Pd = Pd + 1   
                l1 , l2 = 0, 0
                for j in range(args.num_nodes):
                    if Label_tst_f[k+args.lag-1,0,j+1] > 0.5:
                        l1 = l1 + 1
                        if mse01[k,j]>0.5:
                            l2 = l2 + 1
                if l1 == l2:
                    Pi = Pi + 1
    Pf, Pd = Pf/sum(Label_tst_f[args.lag-1:-1,0,0]), Pd/(len(Predict_rnn)-sum(Label_tst_f[args.lag-1:-1,0,0]))
    Pi =  Pi/(len(Predict_rnn)-sum(Label_tst_f[args.lag-1:-1,0,0]))
    PF.append(Pf), PD.append(Pd), PI.append(Pi)
PD = np.array(PD)
PF = np.array(PF)
PI = np.array(PI)

plt.plot(PF,PD,label="Probability of Detection")
plt.plot(PF,PI,label="Probability of Identification")
plt.xscale('log',base=10) 
plt.legend()
plt.show()

plt.savefig('untitled12_mlp__12_12_f5_WT.png')

savetxt(r'/Users/hosseind/Desktop/data/PD_GRN_WT_D_E2_cheb1.csv', PD, delimiter=',')
savetxt(r'/Users/hosseind/Desktop/data/PF_GRN_WT_D_E2_cheb1.csv', PF, delimiter=',')
savetxt(r'/Users/hosseind/Desktop/data/PI_GRN_WT_D_E2_cheb1.csv', PI, delimiter=',')

#plt.plot(data[10*288:288*24,11,0])
#data1 = scaler.inverse_transform(data)
#savetxt(r'/Users/hosseind/Desktop/data/PeMSD8.csv', data1[:,:,0], delimiter=',')


# for case of simple thresholding classification
PD = []
PF = []
PI = []
eps, ep10 = -0.01, 0
while eps < 1:
    eps = eps + 0.001
    if eps > ep10:
        print(eps)
        ep10 = ep10 + 0.1
    mse01 = np.where(res_tst_f<eps, 0, 1) 
    #False alarm and detection
    Pf, Pd, Pi = 0, 0, 0
    for k in range(len(res_tst_f)):
        if (Label_tst_f[k+args.lag-1,0,0] > 0.5):
            if sum(mse01[k,args.lag-1,:])>0.5:
                Pf = Pf + 1
        else:
            if sum(mse01[k,args.lag-1,:])>0.5:
                Pd = Pd + 1   
                l1 , l2 = 0, 0
                for j in range(args.num_nodes):
                    if Label_tst_f[k+args.lag-1,0,j+1] > 0.5:
                        l1 = l1 + 1
                        if mse01[k,args.lag-1,j]>0.5:
                            l2 = l2 + 1
                if l1 == l2:
                    Pi = Pi + 1
    Pf, Pd = Pf/sum(Label_tst_f[args.lag-1:-1,0,0]), Pd/(len(res_tst_f)-sum(Label_tst_f[args.lag-1:-1,0,0]))
    Pi =  Pi/(len(res_tst_f)-sum(Label_tst_f[args.lag-1:-1,0,0]))
    PF.append(Pf), PD.append(Pd), PI.append(Pi)
PD = np.array(PD)
PF = np.array(PF)
PI = np.array(PI)

plt.plot(PF,PD,label="Probability of Detection")
plt.plot(PF,PI,label="Probability of Identification")
plt.xscale('log',base=10) 
plt.legend()
plt.show()




# for the confusion matrix
from sklearn import metrics
import matplotlib as mpl

Predict_rnn_val = history.model.predict(np.flip(res_val_f,1).reshape(-1,args.lag*args.num_nodes))

i=63
plt.plot(range(len(Label_val_f[args.lag-1:-1,0,i+1])),Label_val_f[args.lag-1:-1,0,i+1],label="Label")
plt.plot(range(len(Predict_rnn_val[:,i])),Predict_rnn_val[:,i], label="Predict_RNN")
plt.legend()
plt.show()

PD = []
PF = []
eps, ep10 = -0.01, 0
while eps < 1:
    eps = eps + 0.001
    if eps > ep10:
        print(eps)
        ep10 = ep10 + 0.1
    mse01 = np.where(Predict_rnn_val<eps, 0, 1) 
    #False alarm and detection
    Pf, Pd = 0, 0
    for k in range(len(Predict_rnn_val)):
        if (Label_val_f[k+args.lag-1,0,0] > 0.5):
            if sum(mse01[k,:])>0.5:
                Pf = Pf + 1
        else:
            if sum(mse01[k,:])>0.5:
                Pd = Pd + 1                
    Pf, Pd = Pf/sum(Label_val_f[args.lag-1:-1,0,0]), Pd/(len(Predict_rnn_val)-sum(Label_val_f[args.lag-1:-1,0,0]))
    PF.append(Pf), PD.append(Pd)
PD = np.array(PD)
PF = np.array(PF)

ind = np.argmax(PD-PF) # Youden Index

PD = []
PF = []
eps, ep10 = -0.01, 0
while eps < 1:
    eps = eps + 0.001
    if eps > ep10:
        print(eps)
        ep10 = ep10 + 0.1
    mse01 = np.where(Predict_rnn<eps, 0, 1) 
    #False alarm and detection
    Pf, Pd = 0, 0
    for k in range(len(Predict_rnn)):
        if (Label_tst_f[k+args.lag-1,0,0] > 0.5):
            if sum(mse01[k,:])>0.5:
                Pf = Pf + 1
        else:
            if sum(mse01[k,:])>0.5:
                Pd = Pd + 1                
    Pf, Pd = Pf/sum(Label_tst_f[args.lag-1:-1,0,0]), Pd/(len(Predict_rnn)-sum(Label_tst_f[args.lag-1:-1,0,0]))
    PF.append(Pf), PD.append(Pd)
PD = np.array(PD)
PF = np.array(PF)

plt.plot(PF,PD,label="Probability of Detection")
plt.xscale('log',base=10) 
plt.legend()
plt.show()


PD[ind]
PF[ind]


mse1 = np.zeros(len(Predict_rnn))
eps = -0.01 + 620*0.001
for i in range(len(Predict_rnn)):
    inx = np.argmax(Predict_rnn[i,:])
    if Predict_rnn[i,inx]>eps:
        mse1[i] = inx+1
Label_tstc = np.argmax(Label_tst_f[args.lag-1:-1,0,:], axis=1, out=None)

cmm = metrics.confusion_matrix(Label_tstc, mse1, normalize='all')
savetxt(r'/Users/hosseind/Desktop/data/cmm.csv', cmm, delimiter=',')
savetxt(r'/Users/hosseind/Desktop/data/true.csv', Label_tstc, delimiter=',')
savetxt(r'/Users/hosseind/Desktop/data/predicted.csv', mse1, delimiter=',')

'''
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmm)
plt.title('Confusion matrix of the classifier')
#fig.colorbar(cax)
cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
hsv_modified = cm.get_cmap('hsv', 256)# create new hsv colormaps in range of 0.3 (green) to 0.7 (blue)
newcmp = ListedColormap(hsv_modified(np.linspace(0.3, 0.7, 256)))# show figure
#top = cm.get_cmap('gist_earth', 128) # r means reversed version
#bottom = cm.get_cmap('hsv', 128)# combine it all
#newcolors = np.vstack((top(np.linspace(0, 1, 128)),bottom(np.linspace(0, 1, 128))))
im = ax.imshow(cmm, cmap='hsv')
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cmm)
disp.plot()
'''


'''
#init model
model = AGCRNC(args)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False, weather=False, single=False)

#init loss function, optimizer
loss = nn.CrossEntropyLoss().to(args.device)
loss = torch.nn.MSELoss().to(args.device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
#current_dir = os.path.dirname(os.path.realpath('//home.ansatt.ntnu.no/hosseind/Desktop/AGCRN-master'))
#log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = '/Users/hosseind/Desktop/AGCRN-master/experiment/' + args.dataset


res_f = np.append(res_tr_f, res_val_f, axis=0)
Label_f = np.append(Label_tr_f[args.lag-1:-1,:,:], Label_val_f[args.lag-1:-1,:,:], axis=0)

# RNN
res_f = np.reshape(res_f, (res_f.shape[0], res_f.shape[1], -1)) # (samples, window_size, number_of_features)
res_tst_f = np.reshape(res_tst_f, (res_tst_f.shape[0], res_tst_f.shape[1], -1))

a = np.flip(res_f, 1)

# create repeatitive sets #2
a = np.append(a, a, axis=0)
Label_f = np.append(Label_f, Label_f, axis=0)

a = np.expand_dims(a, axis=3)

B = a[:int(len(a)*0.8),:,:,:]
BL = Label_f[:int(len(a)*0.8),:,:]
C = a[int(len(a)*0.8):,:,:,:]   
CL = Label_f[int(len(a)*0.8):,:]

##############get dataloader######################
train_dataloader_f = data_loader(B, BL, args.batch_size, shuffle=False, drop_last=False)
val_dataloader_f = data_loader(C, CL, args.batch_size, shuffle=False, drop_last=False)
test_dataloader_f = data_loader(np.expand_dims(res_tst_f, axis=3), Label_tst_f[args.lag-1:-1,:,:], args.batch_size, shuffle=False, drop_last=False)
    

#start training
trainer = Trainer(model, loss, optimizer, train_dataloader_f, val_dataloader_f, test_dataloader_f, scaler,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'Train':
    y_true, y_pred = trainer.train()
elif args.mode == 'Test':
    model.load_state_dict(torch.load('/Users/hosseind/Desktop/AGCRN-master/experiment/' + args.dataset + '/' + 'best_model.pth'))
    print("Load saved model")
    y_true, y_pred = trainer.test(model, trainer.args, test_dataloader_f, scaler, trainer.logger)
else:
    raise ValueError
'''













#init model
model2 = AGCRN_Class(args)
model2 = model2.to(args.device)
for p in model2.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model2, only_num=False)


#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model2.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
                    
#start training classifier
trainer = Trainer(model2, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'Train':
    y_true, y_pred = trainer.train()
elif args.mode == 'Test':
    model2.load_state_dict(torch.load('/Users/hosseind/Desktop/AGCRN-master/experiment/' + args.dataset + '/' + 'best_model.pth'))
    print("Load saved model")
    y_true, y_pred = trainer.test(model2, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError

#%%

# creates bias fault and label (for 'max01' normalizerr)
def BiasFaultPM(Nf,N,Nn,sn, m, x,size):
    # x = minimum space between two starting fault is x
    # Nf = No. of simultanuous faults
    # N = No. of faults per sensor (No of Total Faults are N*sn)
    # Nn = N-Nn > No. of faults > N+Nn
    # m = faults length
    BiasFt = np.zeros((size,sn),dtype='float32')
    Labelt = np.zeros((size,sn+1),dtype='int16')
    size = round(size/sn)-1
    Sn = np.array(range(sn))
    for k in range(sn):
        BiasF = np.zeros((size,sn),dtype='float32')
        Label = np.zeros((size,sn+1),dtype='float32')  
        
        rng_ind = np.array(range(size))
        rng_ind = np.reshape(rng_ind, (-1,1))
        for i in range(sn-1):
            rng_ind = np.append(rng_ind,np.reshape(np.array(range(size)), (-1,1)), axis=1)
        for i in range(np.random.randint(N-Nn,N+Nn+1)):
            Sn2 = np.array(range(sn))
            j = random.choice(Sn)
            if np.array(np.nonzero(rng_ind[:,j])).size > 0:
                index = np.random.choice(np.reshape(np.nonzero(rng_ind[:,j]),(-1,)))
                if index > x and index < size-m-x: # minimum space between two starting fault is x
                    rng_ind[index-x:index+x+m,:] = 0
                elif index > size-x-m:
                    rng_ind[index-x-m:size,:] = 0
                    index = size-x-m
                else:
                    rng_ind[0:index+x+m,:] = 0
                    index = x
                rnd = np.random.randint(3,m) # range is 3 to m-1
                BiasF[index:index+rnd,j] = (np.ones(rnd))*random.uniform((0.2), (0.401))*((2*np.random.randint(0,2))-1)
                Label[index:index+rnd,j+1] = (np.ones(rnd))
                if Nf > 1:
                    for l in range(Nf-1):
                        Sn2 = np.delete(Sn2, np.where(Sn2 == j))
                        j = np.random.choice(Sn2)
                        if index > max(x,np.ceil(m/2).astype('int16')) and index < size-np.ceil(x*1.5).astype('int16')-m: # minimum space between two starting fault is x
                            index = np.random.choice(range(index-np.ceil(m/2).astype('int16'),index+np.ceil(m/2).astype('int16')))
                            rng_ind[index-x:index+x+m,:] = 0
                        elif index > size-np.ceil(x*1.5).astype('int16')-m:
                            index = np.random.choice(range(size-np.ceil(m*1.5).astype('int16'),size-m-1))
                            rng_ind[size-np.ceil(m*1.5).astype('int16'):size,:] = 0
                        else:
                            index = np.random.choice(range(0,np.ceil(m/2).astype('int16')))
                            rng_ind[0:index+np.ceil(m/2).astype('int16'),:] = 0
                        rnd = np.random.randint(3,m) # range is 3 to m-1
                        BiasF[index:index+rnd,j] = (np.ones(rnd))*random.uniform((0.2), (0.401))*((2*np.random.randint(0,2))-1)
                        Label[index:index+rnd,j+1] = (np.ones(rnd))              
        a ,b = k*size, (k+1)*size
        BiasFt[a:b,:] = BiasF
        Label[np.where(~Label.any(axis=1))[0],0] = 1
        Labelt[a:b,:] = Label
    return BiasFt, Labelt  



# creates bias fault and label
def NoiseFaultPM(Nf,N,Nn,sn, m, x,size):
    BiasFt = np.zeros((size,sn),dtype='float32')
    Labelt = np.zeros((size,sn+1),dtype='int16')
    size = round(size/sn)-1
    Sn = np.array(range(sn))
    for k in range(sn):
        BiasF = np.zeros((size,sn),dtype='float32')
        Label = np.zeros((size,sn+1),dtype='float32')  
        
        rng_ind = np.array(range(size))
        rng_ind = np.reshape(rng_ind, (-1,1))
        for i in range(sn-1):
            rng_ind = np.append(rng_ind,np.reshape(np.array(range(size)), (-1,1)), axis=1)
        for i in range(np.random.randint(N-Nn,N+Nn+1)):
            Sn2 = np.array(range(sn))
            j = random.choice(Sn)
            if np.array(np.nonzero(rng_ind[:,j])).size > 0:
                index = random.choice(np.reshape(np.nonzero(rng_ind[:,j]),(-1,)))
                if index > x and index < size-m-x: # minimum space between two starting fault is x
                    rng_ind[index-x:index+x+m,:] = 0
                elif index > size-x-m:
                    rng_ind[index-x-m:size,:] = 0
                    index = size-x-m
                else:
                    rng_ind[0:index+x+m,:] = 0
                    index = x
                rnd = random.randint(3,m) # range is 3 to m
                BiasF[index:index+rnd,j] = (np.random.randn(rnd))*random.uniform(0.2, 0.401)
                Label[index:index+rnd,j+1] = (np.ones(rnd))
                if Nf > 1:
                    for l in range(Nf-1):
                        Sn2 = np.delete(Sn2, np.where(Sn2 == j))
                        j = random.choice(np.delete(Sn,j))
                        if index > max(x,np.ceil(m/2).astype('int16')) and index < size-np.ceil(x*1.5).astype('int16')-m: # minimum space between two starting fault is x
                            index = np.random.choice(range(index-np.ceil(m/2).astype('int16'),index+np.ceil(m/2).astype('int16')))
                            rng_ind[index-x:index+x+m,:] = 0
                        elif index > size-np.ceil(x*1.5).astype('int16')-m:
                            index = np.random.choice(range(size-np.ceil(m*1.5).astype('int16'),size-m-1))
                            rng_ind[size-np.ceil(m*1.5).astype('int16'):size,:] = 0
                        else:
                            index = np.random.choice(range(0,np.ceil(m/2).astype('int16')))
                            rng_ind[0:index+np.ceil(m/2).astype('int16'),:] = 0
                        rnd = random.randint(3,m) # range is 3 to m-1
                        BiasF[index:index+rnd,j] = (np.random.randn(rnd))*random.uniform(0.2, 0.401)
                        Label[index:index+rnd,j+1] = (np.ones(rnd))              
        a ,b = k*size, (k+1)*size
        BiasFt[a:b,:] = BiasF
        Label[np.where(~Label.any(axis=1))[0],0] = 1
        Labelt[a:b,:] = Label
    return BiasFt, Labelt     




# creates drift fault and label (for 'max01' normalizerr)
def DriftFaultPM(Nf,N,Nn,sn, m, x,size):
    # x = minimum space between two starting fault is x
    DriftFt = np.zeros((size,sn),dtype='float32')
    Labelt = np.zeros((size,sn+1),dtype='int16')
    size = round(size/sn)-1
    Sn = np.array(range(sn))
    for k in range(sn):
        DriftF = np.zeros((size,sn),dtype='float32')
        Label = np.zeros((size,sn+1),dtype='float32')
        
        rng_ind = np.array(range(size))
        rng_ind = np.reshape(rng_ind, (-1,1))
        for i in range(sn-1):
            rng_ind = np.append(rng_ind,np.reshape(np.array(range(size)), (-1,1)), axis=1)
        for i in range(np.random.randint(N-Nn,N+Nn+1)):
            Sn2 = np.array(range(sn))
            j = random.choice(Sn)
            if np.array(np.nonzero(rng_ind[:,j])).size > 0:
                index = np.random.choice(np.reshape(np.nonzero(rng_ind[:,j]),(-1,)))
                if index > x and index < size-m-x: # minimum space between two starting fault is x
                    rng_ind[index-x:index+m+x,:] = 0
                elif index > size-x-m:
                    rng_ind[index-x-m:size,:] = 0
                    index = size-x-m
                else:
                    rng_ind[0:index+x+m,:] = 0
                    index = x
                rnd = np.random.randint(4,m) # range is 4 to m-1
                rndD = round((0.6)*rnd) # 0.6 of fault is drift and the rest is bias
                rndDrift = (1/(rndD))*(np.arange(1,rndD+1))
                rndBias = np.ones(rnd-rndD)
                rndF = np.append(rndDrift,rndBias)
                DriftF[index:index+rnd,j] =  rndF*(random.uniform(0.2, 0.401))*((2*np.random.randint(2))-1)
                Label[index:index+rnd,j+1] = (np.ones(rnd))  # +1 and -1 act as one sample tolerance
                if Nf > 1:
                    for l in range(Nf-1):
                        Sn2 = np.delete(Sn2, np.where(Sn2 == j))
                        j = np.random.choice(Sn2)
                        if index > max(x,np.ceil(m/2).astype('int16')) and index < size-np.ceil(x*1.5).astype('int16')-m: # minimum space between two starting fault is x
                            index = np.random.choice(range(index-np.ceil(m/2).astype('int16'),index+np.ceil(m/2).astype('int16')))
                            rng_ind[index-x:index+x+m,:] = 0
                        elif index > size-np.ceil(x*1.5).astype('int16')-m:
                            index = np.random.choice(range(size-np.ceil(m*1.5).astype('int16'),size-m-1))
                            rng_ind[size-np.ceil(m*1.5).astype('int16'):size,:] = 0
                        else:
                            index = np.random.choice(range(0,np.ceil(m/2).astype('int16')))
                            rng_ind[0:index+np.ceil(m/2).astype('int16'),:] = 0
                        rnd = np.random.randint(4,m) # range is 4 to m-1
                        rndD = round((0.6)*rnd) # 0.6 of fault is drift and the rest is bias
                        rndDrift = (1/(rndD))*(np.arange(1,rndD+1))
                        rndBias = np.ones(rnd-rndD)
                        rndF = np.append(rndDrift,rndBias)
                        DriftF[index:index+rnd,j] =  rndF*(random.uniform(0.2, 0.401))*((2*np.random.randint(2))-1)
                        Label[index:index+rnd,j+1] = (np.ones(rnd))  # +1 and -1 act as one sample tolerance
        a ,b = k*size, (k+1)*size
        DriftFt[a:b,:] = DriftF
        Label[np.where(~Label.any(axis=1))[0],0] = 1
        Labelt[a:b,:] = Label
    return DriftFt, Labelt



# creates freeze fault and label (for 'max01' normalizerr)
def FreezeFaultPM(Nf,N,Nn,sn,m1,m,size):
    x = m+1
    # x minimum space between two starting fault is x
    # m1 = minimum faults length
    BiasFt = np.zeros((size,sn),dtype='float32')
    Labelt = np.zeros((size,sn+1),dtype='int16')
    size = round(size/sn)-1
    Sn = np.array(range(sn))
    for k in range(sn):
        BiasF = np.zeros((size,sn),dtype='float32')
        Label = np.zeros((size,sn+1),dtype='float32')
        
        rng_ind = np.array(range(size))
        rng_ind = np.reshape(rng_ind, (-1,1))
        for i in range(sn-1):
            rng_ind = np.append(rng_ind,np.reshape(np.array(range(size)), (-1,1)), axis=1)
        for i in range(np.random.randint(N-Nn,N+Nn+1)):
            Sn2 = np.array(range(sn))
            j = random.choice(Sn)
            if np.array(np.nonzero(rng_ind[:,j])).size > 0:    
                index = random.choice(np.reshape(np.nonzero(rng_ind[:,j]),(-1,)))
                if index > x and index < size-m-x: # minimum space between two starting fault is x
                    rng_ind[index-x:index+m+x,:] = 0
                elif index > size-x-m:
                    rng_ind[index-x-m:size,:] = 0
                    index = size-x-m
                else:
                    rng_ind[0:index+x+m,:] = 0
                    index = x 
                rnd = random.randint(m1,m) # range is m1 to m
                BiasF[index:index+rnd,j] = (np.ones(rnd))
                Label[index:index+rnd,j+1] = (np.ones(rnd))
                if Nf > 1:
                    for l in range(Nf-1):
                        Sn2 = np.delete(Sn2, np.where(Sn2 == j))
                        j = random.choice(np.delete(Sn,j))
                        if index > max(x,np.ceil(m/2).astype('int16')) and index < size-np.ceil(x*1.5).astype('int16')-m: # minimum space between two starting fault is x
                            index = np.random.choice(range(index-np.ceil(m/2).astype('int16'),index+np.ceil(m/2).astype('int16')))
                            rng_ind[index-x:index+x+m,:] = 0
                        elif index > size-np.ceil(x*1.5).astype('int16')-m:
                            index = np.random.choice(range(size-np.ceil(m*1.5).astype('int16'),size-m-1))
                            rng_ind[size-np.ceil(m*1.5).astype('int16'):size,:] = 0
                        else:
                            index = np.random.choice(range(0,np.ceil(m/2).astype('int16')))
                            rng_ind[0:index+np.ceil(m/2).astype('int16'),:] = 0
                        rnd = random.randint(m1,m) # range is m1 to m
                        BiasF[index:index+rnd,j] = (np.ones(rnd))
                        Label[index:index+rnd,j+1] = (np.ones(rnd))              
        a ,b = k*size, (k+1)*size
        BiasFt[a:b,:] = BiasF
        Label[np.where(~Label.any(axis=1))[0],0] = 1
        Labelt[a:b,:] = Label
    return BiasFt, Labelt   

      


# creates bias fault and label (for 'std' normalizer)
def BiasFaultPMstd(Nf,N,Nn,sn, m, x,size):
    # x = minimum space between two starting fault is x
    BiasFt = np.zeros((size,sn),dtype='float32')
    Labelt = np.zeros((size,sn+1),dtype='int8')
    size = round(size/sn)-1
    Sn = np.array(range(sn))
    for k in range(sn):
        BiasF = np.zeros((size,sn),dtype='float32')
        Label = np.zeros((size,sn+1),dtype='float32')  
        rng_ind = np.array(range(size))
        
        rng_ind = np.reshape(rng_ind, (-1,1))
        for i in range(sn-1):
            rng_ind = np.append(rng_ind,np.reshape(np.array(range(size)), (-1,1)), axis=1)
        for i in range(np.random.randint(N-Nn,N+Nn+1)):
            Sn2 = np.array(range(sn))
            j = random.choice(Sn)
            if np.array(np.nonzero(rng_ind[:,j])).size > 0:
                index = np.random.choice(np.reshape(np.nonzero(rng_ind[:,j]),(-1,)))
                if index > x and index < size-m-x: # minimum space between two starting fault is x
                    rng_ind[index-x:index+x+m,:] = 0
                elif index > size-x-m:
                    rng_ind[index-x-m:size,:] = 0
                    index = size-x-m
                else:
                    rng_ind[0:index+x+m,:] = 0
                    index = x
                rnd = np.random.randint(3,m) # range is 3 to m-1
                BiasF[index:index+rnd,j] = (np.ones(rnd))*random.uniform(np.sqrt(0.2), np.sqrt(0.401))*((2*np.random.randint(0,2))-1)
                Label[index:index+rnd,j+1] = (np.ones(rnd))
                if Nf > 1:
                    for l in range(Nf-1):
                        Sn2 = np.delete(Sn2, np.where(Sn2 == j))
                        j = np.random.choice(Sn2)
                        if index > max(x,np.ceil(m/2).astype('int8')) and index < size-np.ceil(x*1.5).astype('int8')-m: # minimum space between two starting fault is x
                            index = np.random.choice(range(index-np.ceil(m/2).astype('int8'),index+np.ceil(m/2).astype('int8')))
                            rng_ind[index-x:index+x+m,:] = 0
                        elif index > size-np.ceil(x*1.5).astype('int8')-m:
                            index = np.random.choice(range(size-np.ceil(m*1.5).astype('int8'),size-m-1))
                            rng_ind[size-np.ceil(m*1.5).astype('int8'):size,:] = 0
                        else:
                            index = np.random.choice(range(0,np.ceil(m/2).astype('int8')))
                            rng_ind[0:index+np.ceil(m/2).astype('int8'),:] = 0
                        rnd = np.random.randint(3,m) # range is 3 to m-1
                        BiasF[index:index+rnd,j] = (np.ones(rnd))*random.uniform(np.sqrt(0.2), np.sqrt(0.401))*((2*np.random.randint(0,2))-1)
                        Label[index:index+rnd,j+1] = (np.ones(rnd))              
        a ,b = k*size, (k+1)*size
        BiasFt[a:b,:] = BiasF
        Label[np.where(~Label.any(axis=1))[0],0] = 1
        Labelt[a:b,:] = Label
    return BiasFt, Labelt  



#%%
# load the dataset
dataframe = pd.read_csv(r'/Users/hosseind/Downloads/785decaf9bc7319862bfe9fe83cd33c1/1270825_43.19_-96.47_2012.csv')
dataset1 = dataframe.values

dataset2 = np.append(dataset2,dataset1[1:,5].reshape(-1,1), axis=1)

#dataset[:,18] = dataset1[1:,5]

#dataset = np.zeros((8760,200))




a = np.append(a, a, axis=0)
Label_f = np.append(Label_f, Label_f, axis=0)


# fit RNN model
with tf.device('/CPU:0'):
    history = RNN(a, Label_f[:,0,1:])


from torch.utils.data import TensorDataset, DataLoader
import time

batch_size = 64
train_datarnn = TensorDataset(torch.from_numpy(a), torch.from_numpy(Label_f[:,0,1:]))
train_loaderrnn = DataLoader(train_datarnn, shuffle=True, batch_size=batch_size, drop_last=True)



class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cpu')
        return hidden



def train(train_loaderrnn, learn_rate, hidden_dim=100, EPOCHS=5, model_type="GRU"):
    
    # Setting common hyperparameters
    input_dim = next(iter(train_loaderrnn))[0].shape[2]
    output_dim = 1
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to('cpu')
    
    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loaderrnn:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()
            
            out, h = model(x.to('cpu').float(), h)
            loss = criterion(out, label.to('cpu').float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
    return model

lr = 0.001
gru_model = train(train_loaderrnn, lr, model_type="GRU")


def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
    print("sMAPE: {}%".format(sMAPE*100))
    return outputs, targets, sMAPE


gru_outputs, targets, gru_sMAPE = evaluate(gru_model, np.flip(res_tst_f,1), Label_tst_f[args.lag-1:-1,0,1:], label_scalers)


plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
plt.plot(gru_outputs[0][-100:], "-o", color="g", label="Predicted")
plt.plot(targets[0][-100:], color="b", label="Actual")
plt.ylabel('Energy Consumption (MW)')
plt.legend()

plt.subplot(2,2,2)
plt.plot(gru_outputs[8][-50:], "-o", color="g", label="Predicted")
plt.plot(targets[8][-50:], color="b", label="Actual")
plt.ylabel('Energy Consumption (MW)')
plt.legend()

plt.subplot(2,2,3)
plt.plot(gru_outputs[4][:50], "-o", color="g", label="Predicted")
plt.plot(targets[4][:50], color="b", label="Actual")
plt.ylabel('Energy Consumption (MW)')
plt.legend()

plt.subplot(2,2,4)
plt.plot(lstm_outputs[6][:100], "-o", color="g", label="Predicted")
plt.plot(targets[6][:100], color="b", label="Actual")
plt.ylabel('Energy Consumption (MW)')
plt.legend()
plt.show()




#%%
import networkx as nx


A = pd.read_csv(r'/Users/hosseind/Desktop/GraphDataset-master/connection_matrix_inverse_distance.csv')
A = A.values
A = A[:,1:] 

G = nx.from_numpy_matrix(A)
nx.draw(G)


























