# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:12:56 2021

@author: hosseind
"""

'''	
from random import random
from numpy import array
from pandas import concat
from pandas import DataFrame
from keras.layers import Masking
# generate a sequence of random values
def generate_sequence(n_timesteps):
	return [random() for _ in range(n_timesteps)]
 
# generate data for the lstm
def generate_data(n_timesteps):
	# generate sequence
	sequence = generate_sequence(n_timesteps)
	sequence = array(sequence)
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	# remove rows with missing values
	##df.dropna(inplace=True)
    # replace missing values with -1
	df.fillna(-1, inplace=True)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 0]
	return X, y
 
# generate sequence
n_timesteps = 10
X, y = generate_data(n_timesteps)

# If all features for a timestep contain the masked value, then the whole timestep will be excluded from calculations.
## model.add(Masking(mask_value=-1, input_shape=(2, 1)))
'''

'''
# convert an array of values into a dataset matrix
def create_dataset(dataset, Lv=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-Lv-1):
        a = dataset[i:(i+Lv), 0]
        dataX.append(a)
        dataY.append(dataset[i + Lv, 0])
    return np.array(dataX), np.array(dataY)
'''
#%%
import os
import copy
from skimage.measure import block_reduce
import sklearn.preprocessing
import sklearn.impute
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
np. __version__ 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
from numpy import savetxt
from numpy import random

import tensorflow as tf
print(tf.__version__)
from tensorflow import keras


# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t, t-1, t-2)
import math
from keras.models import Sequential
from keras.layers import Dense, Input, LeakyReLU, Dropout, GRU, LSTM, Bidirectional
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.models import load_model, Model
from keras import optimizers
#from keras.utils import plot_model

# To reproduce the results
#from numpy.random import seed
#seed(1)
#keras.backend.clear_session()
#np.random.seed(42)
#tf.random.set_seed(42)


# convert an matrix of values into a delayed matrix
def create_dataset(dataset, Lv=0):
    data = []
    for i in range(len(dataset[1,:])):
        for k in range(Lv+1):
            if k==0:
                a = dataset[Lv-k:, i]
            else:
                a = dataset[Lv-k:-k, i]       
            data.append(a)
    return np.array(data)


# fit virtual sensor AE
def vs(x,y):
    # create and fit Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(p, input_dim=(sn+rn), activation='sigmoid'))
    model.add(Dense(3, input_dim=(sn+rn), activation='sigmoid'))
    model.add(Dense(p, input_dim=(sn+rn), activation='sigmoid'))
    ## model.add(LeakyReLU(alpha=0.1))
    #model.add(Dropout(0.0))
    #model.add(Dense(p, activation='tanh'))
    model.add(Dense((sn+rn)))
    optim = keras.optimizers.Nadam(learning_rate=l, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer=optim, metrics=['mse'])
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, verbose=1, patience=20, restore_best_weights=True)
    #mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # fit model
    hist = model.fit(x,y, epochs=400, batch_size=bs, verbose=1, 
           validation_split=0.15, shuffle=False, callbacks=[es])
    return hist

# fit virtual sensor AE attack
def vs2(x,y):
    # create and fit Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(69, input_dim=((sn+rn)*(Lv+1)), activation='sigmoid'))
    model.add(Dense(49, activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(49, activation='sigmoid'))
    model.add(Dense(69, activation='sigmoid'))
    model.add(Dense(((sn+rn)*(Lv+1)), activation='sigmoid'))
    ## model.add(LeakyReLU(alpha=0.1))
    #model.add(Dropout(0.0))
    #model.add(Dense(p, activation='tanh'))
    optim = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optim, metrics=['mse'])
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, verbose=1, patience=20, restore_best_weights=True)
    #mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # fit model
    hist = model.fit(x,y, epochs=400, batch_size=bs, verbose=1, 
           validation_split=0.15, shuffle=False, callbacks=[es])
    return hist

# fit virtual sensor AE attack PemsD8
def vsp8(x,y):
    # create and fit Multilayer Perceptron model
    ''' model.add(Dense(4231, input_dim=((sn+rn)*(Lv+1)), activation='sigmoid'))
     model.add(Dense(3022, activation='sigmoid'))
     model.add(Dense(1813, activation='sigmoid'))
     model.add(Dense(3022, activation='sigmoid'))
     model.add(Dense(4231, activation='sigmoid'))'''
    
    model = Sequential()
    model.add(Dense(900, input_dim=((sn+rn)*(Lv+1)), activation='sigmoid'))
    model.add(Dense(600, activation='sigmoid'))
    model.add(Dense(300, activation='sigmoid'))
    model.add(Dense(600, activation='sigmoid'))
    model.add(Dense(900, activation='sigmoid'))
    model.add(Dense(((sn+rn)*(Lv+1)), activation='sigmoid'))
    ## model.add(LeakyReLU(alpha=0.1))
    #model.add(Dropout(0.0))
    #model.add(Dense(p, activation='tanh'))
    optim = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optim, metrics=['mse'])
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, verbose=1, patience=20, restore_best_weights=True)
    #mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # fit model
    hist = model.fit(x,y, epochs=400, batch_size=bs, verbose=1, 
           validation_split=0.25, shuffle=False, callbacks=[es])
    return hist


# fit virtual sensor AE attack (for eusipco 2022)
def vs3(x,y):
    # create and fit Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(93, input_dim=((sn+rn)*(Lv+1)), activation='sigmoid'))
    model.add(Dense(66, activation='sigmoid'))
    model.add(Dense(40, activation='sigmoid'))
    model.add(Dense(66, activation='sigmoid'))
    model.add(Dense(93, activation='sigmoid'))
    model.add(Dense(((sn+rn)*(Lv+1)), activation='sigmoid'))
    ## model.add(LeakyReLU(alpha=0.1))
    #model.add(Dropout(0.0))
    #model.add(Dense(p, activation='tanh'))
    optim = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optim, metrics=['mse'])
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, verbose=1, patience=20, restore_best_weights=True)
    #mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # fit model
    hist = model.fit(x,y, epochs=400, batch_size=bs, verbose=1, 
           validation_split=0.15, shuffle=False, callbacks=[es])
    return hist

# creates bias fault and label
def BiasFault(Nf,N,Nn,size):
    x = 10 # minimum space between two starting fault is x
    BiasFt = np.zeros((sn*size,sn),dtype=d_type)
    Labelt = np.zeros((sn*size,sn+1),dtype='int8')
    for k in range(sn):
        BiasF = np.zeros((size,sn),dtype=d_type)
        Label = np.zeros((size,sn+1),dtype=d_type)
        Sn = np.array(range(sn))
        rng_ind = np.array(range(size))
        rng_ind = np.reshape(rng_ind, (-1,1))
        for i in range(sn-1):
            rng_ind = np.append(rng_ind,np.reshape(np.array(range(size)), (-1,1)), axis=1)
        for i in range(np.random.randint(N-Nn,N+Nn+1)):
            j = random.choice(Sn)
            index = random.choice(np.reshape(np.nonzero(rng_ind[:,j]),(-1,)))
            if index > x and index < size-x: # minimum space between two starting fault is x
                rng_ind[index-x:index+x,:] = 0
            elif index > size-x:
                rng_ind[index-x:size,:] = 0
                index = size-x
            else:
                rng_ind[0:index+x,:] = 0
                index = 0
            rnd = random.randint(3,8) # range is 3 to 7
            BiasF[index:index+rnd,j] = (np.ones(rnd))*random.uniform(0.2, 0.401)*((2*random.randint(2))-1)
            Label[index:index+rnd,j+1] = (np.ones(rnd))
            if Nf > 1:
                for l in range(Nf-1):
                    j = random.choice(np.delete(Sn,j))
                    if index > x and index < size-np.ceil(x*1.5).astype('int8'): # minimum space between two starting fault is x
                        index = random.choice(range(index-np.ceil(x/2).astype('int8'),index+np.ceil(x/2).astype('int8')))
                        rng_ind[index-x:index+x,:] = 0
                    elif index > size-np.ceil(x*1.5).astype('int8'):
                        index = random.choice(range(size-np.ceil(x*1.5).astype('int8'),size-x))
                        rng_ind[size-np.ceil(x*1.5).astype('int8'):size,:] = 0
                    else:
                        index = random.choice(range(0,np.ceil(x/2).astype('int8')))
                        rng_ind[0:index+np.ceil(x/2).astype('int8'),:] = 0
                    rnd = random.randint(3,8) # range is 3 to 7
                    BiasF[index:index+rnd,j] = (np.ones(rnd))*random.uniform(0.2, 0.401)*((2*random.randint(2))-1)
                    Label[index:index+rnd,j+1] = (np.ones(rnd))              
        a ,b = k*size, (k+1)*size
        BiasFt[a:b,:] = BiasF
        Label[np.where(~Label.any(axis=1))[0],0] = 1
        Labelt[a:b,:] = Label
    return BiasFt, Labelt            


# creates drift fault and label
def DriftFault(Nf,N,Nn,size):
    x = 13 # minimum space between two starting fault is x
    DriftFt = np.zeros((sn*size,sn),dtype=d_type)
    Labelt = np.zeros((sn*size,sn+1),dtype='int8')
    for k in range(sn):
        DriftF = np.zeros((size,sn),dtype=d_type)
        Label = np.zeros((size,sn+1),dtype=d_type)
        Sn = np.array(range(sn))
        rng_ind = np.array(range(size))
        rng_ind = np.reshape(rng_ind, (-1,1))
        for i in range(sn-1):
            rng_ind = np.append(rng_ind,np.reshape(np.array(range(size)), (-1,1)), axis=1)
        for i in range(np.random.randint(N-Nn,N+Nn+1)):
            j = random.choice(Sn)
            index = random.choice(np.reshape(np.nonzero(rng_ind[:,j]),(-1,)))
            if index > x and index < size-x: # minimum space between two starting fault is x
                rng_ind[index-x:index+x,:] = 0
            elif index > size-x:
                rng_ind[index-x:size,:] = 0
                index = size-x
            else:
                rng_ind[0:index+x,:] = 0
                index = 0
            rnd = random.randint(4,12) # range is 4 to 11
            rndD = round((0.6)*rnd) # 0.6 of fault is drift and the rest is bias
            rndDrift = (1/(rndD))*(np.arange(1,rndD+1))
            rndBias = np.ones(rnd-rndD)
            rndF = np.append(rndDrift,rndBias)
            DriftF[index:index+rnd,j] =  rndF*(random.uniform(0.2, 0.401))*((2*random.randint(2))-1)
            Label[index:index+rnd,j+1] = (np.ones(rnd))  # +1 and -1 act as one sample tolerance
            if Nf > 1:
                for l in range(Nf-1):
                    j = random.choice(np.delete(Sn,j))
                    if index > x and index < size-np.ceil(x*1.5).astype('int8'): # minimum space between two starting fault is x
                        index = random.choice(range(index-np.ceil(x/2).astype('int8'),index+np.ceil(x/2).astype('int8')))
                        rng_ind[index-x:index+x,:] = 0
                    elif index > size-np.ceil(x*1.5).astype('int8'):
                        index = random.choice(range(size-np.ceil(x*1.5).astype('int8'),size-x))
                        rng_ind[size-np.ceil(x*1.5).astype('int8'):size,:] = 0
                    else:
                        index = random.choice(range(0,np.ceil(x/2).astype('int8')))
                        rng_ind[0:index+np.ceil(x/2).astype('int8'),:] = 0
                    rnd = random.randint(4,12) # range is 4 to 11
                    rndD = round((0.6)*rnd) # 0.6 of fault is drift and the rest is bias
                    rndDrift = (1/(rndD))*(np.arange(1,rndD+1))
                    rndBias = np.ones(rnd-rndD)
                    rndF = np.append(rndDrift,rndBias)
                    DriftF[index:index+rnd,j] =  rndF*(random.uniform(0.2, 0.401))*((2*random.randint(2))-1)
                    Label[index:index+rnd,j+1] = (np.ones(rnd))  # +1 and -1 act as one sample tolerance
        a ,b = k*size, (k+1)*size
        DriftFt[a:b,:] = DriftF
        Label[np.where(~Label.any(axis=1))[0],0] = 1
        Labelt[a:b,:] = Label
    return DriftFt, Labelt    


#%%
# load the dataset
dataframe = pd.read_excel(r'/Users/hosseind/Desktop/AirQualityUCI.xlsx')
dataset = dataframe.values
M=5
dataset = dataset[M:-M,:]
dataset = np.delete(dataset,[0,1,2,4,5,7,9,14],1)
sn, rn = 5 ,2 # rn measurments should be located at the last columns
p , pc , p2 = 10, 15, 10
Lv = 10
Lc = 10
Lp = 10
bs , l = 20, 0.0004
d_type='float32'

# preprocessing
dataset = dataset.astype(d_type)
dataset = np.where(dataset==-200, np.NaN, dataset) 

dataframe = pd.DataFrame(dataset)
dataframe.dropna(inplace=True)
dataset = dataframe.values

##imp = sklearn.impute.KNNImputer(n_neighbors=4)
##dataset = imp.fit_transform(dataset)
df = pd.DataFrame(dataset)
df.corr()

#%%
# load the dataset
dataframe = pd.read_csv(r'/Users/hosseind/Desktop/pmsm_temperature_data.csv')
dataset = dataframe.values
dataset = dataset[::10,:]
plt.plot(dataset[:,12])
plt.plot(dataset[:,3])
dataset = dataset[47900:55850,:]
M=5
dataset = dataset[M:-M,:]
dataset[:,2] = np.add(dataset[:,2], dataset[:,3])
dataset[:,3] = np.add(dataset[:,6], dataset[:,7])
#dataset[:,2] = np.sqrt(np.add(np.power(dataset[:,2],2), np.power(dataset[:,3],2)))
#dataset[:,3] = np.sqrt(np.add(np.power(dataset[:,6],2), np.power(dataset[:,7],2)))
dataset = np.delete(dataset,[0,6,7,8,12],1)
del dataframe
sn, rn = 5 ,3  # rn measurments should be located at the last columns
p , pc , p2 = 10, 15, 10
Lv = 10
Lc = 10
Lp = 10
d_type='float32'
bs , l = 50, 0.001
# preprocessing
dataset = dataset.astype(d_type)
dataset = np.where(dataset<=-150, np.NaN, dataset) 

dataframe = pd.DataFrame(dataset)
dataframe.dropna(inplace=True)
dataset = dataframe.values

df = pd.DataFrame(dataset)
df.corr()
#%%
# load the dataset
dataframe = pd.read_csv(r'\\home.ansatt.ntnu.no\hosseind\Desktop\IntelLab.csv')
dataset = dataframe.values
dataset = dataset[0:2300000,:3]
del dataframe

sn, rn = 5 ,0  # rn measurments should be located at the last columns
p , pc , p2 = 10, 15, 10
Lv = 10
Lc = 10
Lp = 10
d_type='float32'
bs , l = 20, 0.0004
# preprocessing
M=5
dataset = dataset[M:-M,:]
sample = 12000
sample_start = 5000
dataset1 = list(np.where( dataset[:,1] == 9))[0]
dataset1 = dataset[dataset1,2]
dataset2 = list(np.where( dataset[:,1] == 19))[0]
dataset2 = dataset[dataset2,2]
dataset3 = list(np.where( dataset[:,1] == 31))[0]
dataset3 = dataset[dataset3,2]
dataset4 = list(np.where( dataset[:,1] == 40))[0]
dataset4 = dataset[dataset4,2]
dataset5 = list(np.where( dataset[:,1] == 49))[0]
dataset5 = dataset[dataset5,2]
dataset = [dataset1[sample_start:sample],dataset2[sample_start:sample],dataset3[sample_start:sample],dataset4[sample_start:sample],dataset5[sample_start:sample]]
dataset = np.array(dataset)
dataset = np.reshape(dataset, (-1,sn+rn))

dataset = dataset.astype(d_type)
dataset = np.where(dataset<=3, np.NaN, dataset) 

dataframe = pd.DataFrame(dataset)
dataframe.dropna(inplace=True)
dataset = dataframe.values

df = pd.DataFrame(dataset)
df.corr()
#%%
# load the dataset
dataframe = pd.read_csv(r'\\home.ansatt.ntnu.no\hosseind\Desktop\multihubdataset.csv')
dataset = dataframe.values
dataset = dataset[:4689,:]
del dataframe

sn, rn = 4 ,0 # rn measurments should be located at the last columns
p , pc , p2 = 10, 15, 10
Lv = 10
Lc = 10
Lp = 10
d_type='float32'
bs , l = 20, 0.0004

# preprocessing
dataset = dataset.astype(d_type)

for i in range(sn):
    dataset[:,(3*i)+2] = np.where(dataset[:,(3*i)+2]==1, np.NaN, dataset[:,(3*i)+2]) 
    
dataframe = pd.DataFrame(dataset)
dataframe.dropna(inplace=True)
dataset = dataframe.values

M=5
dataset = dataset[M:-M,:]
dataset = np.delete(dataset,[0,2,3,5,6,8,9,11],1)

df = pd.DataFrame(dataset)
df.corr()
#%%
# load the dataset
dataframe = pd.read_csv(r'\\home.ansatt.ntnu.no\hosseind\Desktop\moka_iteration_15.csv')
dataset_temp = dataframe.values
dataset = dataset_temp[::20,1:]
dataframe = pd.read_csv(r'\\home.ansatt.ntnu.no\hosseind\Desktop\moka_iteration_14.csv')
dataset_temp = dataframe.values
dataset = np.append(dataset,dataset_temp[::20,1:], axis=0)
dataframe = pd.read_csv(r'\\home.ansatt.ntnu.no\hosseind\Desktop\moka_iteration_16.csv')
dataset_temp = dataframe.values
dataset = np.append(dataset,dataset_temp[::20,1:], axis=0)
del dataframe
del dataset_temp

sn, rn = 5 ,0  # rn measurments should be located at the last columns
p , pc , p2 = 10, 15, 10
Lv = 10
Lc = 10
Lp = 10
d_type='float32'
bs , l = 20, 0.0004
# preprocessing
M=5
dataset = dataset[M:-M,:]
dataset = np.array(dataset)
dataset = np.delete(dataset,[0,1,2,3,7,],1)

dataset = dataset.astype(d_type)
dataset = np.where(dataset<=-100, np.NaN, dataset) 

dataframe = pd.DataFrame(dataset)
dataframe.dropna(inplace=True)
dataset = dataframe.values

df = pd.DataFrame(dataset)
df.corr()

#%%
# load the dataset
dataframe = pd.read_csv(r'\\home.ansatt.ntnu.no\hosseind\Desktop\San_Diego_Daily_Weather_Data_2014.csv')
dataset_temp = dataframe.values
dataset_temp = np.array(dataset_temp)
dataset_temp = np.delete(dataset_temp,[0,1,6,7,8,9,10,11],1)
dataset_temp = dataset_temp[:150000,:]
dataset = block_reduce(dataset_temp.transpose(), block_size=(1,15), func=np.mean, cval=np.mean(dataset_temp.transpose()))

sn, rn = 5 ,0  # rn measurments should be located at the last columns
p , pc , p2 = 10, 15, 10
Lv = 10
Lc = 10
Lp = 10
d_type='float32'
bs , l = 20, 0.0004
# preprocessing
M=5
dataset = dataset.transpose()[M:-M,:]
del dataframe
del dataset_temp

dataset = dataset.astype(d_type)
dataset = np.where(dataset<=-1000000, np.NaN, dataset) 

dataframe = pd.DataFrame(dataset)
dataframe.dropna(inplace=True)
dataset = dataframe.values

df = pd.DataFrame(dataset)
df.corr()

#%%
data_path = os.path.join('/Users/hosseind/Desktop/AGCRN-master/data/PEMS08/pems08.npz')
dataset = np.load(data_path)['data'][:, :, 0] 

sn, rn = 170 ,0  # rn measurments should be located at the last columns
p , pc , p2 = 10, 15, 10
Lv = 31
Lc = 31
Lp = 31
d_type='float32'
bs , l = 20, 0.0004
# preprocessing
M=5
dataset = dataset[M:-M,:]

#df = pd.DataFrame(dataset)
#df.corr()
#%% watertank
dataset = pd.read_csv(r'/Users/hosseind/Desktop/GraphDataset-master/measurements_1.csv')
dataset = dataset.values
dataset = dataset[:,1:101]   

sn, rn = 100 ,0  # rn measurments should be located at the last columns
p , pc , p2 = 10, 15, 10
Lv = 31
Lc = 31
Lp = 31
d_type='float32'
bs , l = 20, 0.0004
# preprocessing
M=5
dataset = dataset[M:-M,:]

#df = pd.DataFrame(dataset)
#df.corr()

#%% PemsD8
data_path = os.path.join('/Users/hosseind/Desktop/AGCRN-master/data/PEMS08/pems08.npz')
dataset = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data

sn, rn = 170 ,0  # rn measurments should be located at the last columns
p , pc , p2 = 10, 15, 10
Lv = 11
Lc = 5
Lp = 11
bs , l = 20, 0.0004
d_type='float32'
#%% Water Tanks
data = pd.read_csv(r'/Users/hosseind/Desktop/GraphDataset-master/measurements_1.csv')
data2 = pd.read_csv(r'/Users/hosseind/Desktop/GraphDataset-master/measurements_2.csv')
data3 = pd.read_csv(r'/Users/hosseind/Desktop/GraphDataset-master/measurements_3.csv')
data = data.values
data2 = data2.values
data3 = data3.values
data = np.concatenate((data, data2, data3), axis=0)
dataset = data[:,1:101]    

sn, rn = 100 ,0  # rn measurments should be located at the last columns
p , pc , p2 = 10, 15, 10
Lv = 11
Lc = 5
Lp = 11
bs , l = 20, 0.0004
d_type='float32'
#%%

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

##mask = np.random.rand(len(dataset)) < 0.85
##train, test = dataset[mask], dataset[~mask]

# normalization
std_scale = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train)
train = std_scale.transform(train)
test = std_scale.transform(test)


# fit virtal sensor model
np.random.shuffle(train)
with tf.device('/CPU:0'):
    history1 = vs(train, train)






# reshape dataset (injecting Lv delays)
if (Lp-Lv)>0:
    trainD = create_dataset(train[abs(Lv-Lp):,:], Lv)
    testD = create_dataset(test[abs(Lv-Lp):,:], Lv)
else:
    trainD = create_dataset(train, Lv)
    testD = create_dataset(test, Lv)
trainD = trainD.transpose()
testD = testD.transpose()


# fit virtal sensor model
np.random.shuffle(trainD)

with tf.device('/CPU:0'):
    history11 = vs2(trainD, trainD)


tic()
# for pemsd8, WT
with tf.device('/CPU:0'):
    history11 = vsp8(trainD, trainD)   
toc()


testPredict1 = history11.model.predict(testD) 

i = 0
plt.plot(range(len(testD[:,i])),testD[:,i],range(len(testPredict1)),testPredict1[:,i]) 

testPredict11 = np.reshape(testPredict1[:,0], (-1,1))
for i in range(sn-1):
    testPredict11 = np.append(testPredict11,np.reshape(testPredict1[:,(i+1)*(Lv+1)], (-1,1)), axis=1)

i = 50
plt.plot(range(len(test[Lv:,i])),test[Lv:,i],range(len(testPredict11)),testPredict11[:,i]) 

i = 33
mae, rmse, mape = All_Metrics_m(std_scale.inverse_transform(testPredict11)[:,:i], std_scale.inverse_transform(test[Lv:,:])[:,:i], None, None)


#%%
def MAE_np_m(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def RMSE_np_m(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

def MAPE_np_m(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def All_Metrics_m(pred, true, mask1, mask2):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = MAE_np_m(pred, true, mask1)
        rmse = RMSE_np_m(pred, true, mask1)
        mape = MAPE_np_m(pred, true, mask2)
    else:
        raise TypeError
    return mae, rmse, mape 
    
'''history11.model.summary()'''

'''plot_model(history1)'''

'''
# load the saved model
saved_model = load_model('best_model.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(trainD[:, Lv+1:], trainD[:, 0], verbose=0)
_, test_acc = saved_model.evaluate(testD[:, Lv+1:], testD[:, 0], verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
'''

#%%
# Estimate model performance

plt.plot(history5.history['val_loss'])

trainScore = history1.model.evaluate(trainD[:, Lv+1:], trainD[:, 0], verbose=0)
print('Train Score: %.5f MSE (%.5f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = history11.model.evaluate(testD, testD, verbose=0)
print('Test Score: %.5f MSE (%.5f RMSE)' % (testScore[1], math.sqrt(testScore[1])))


# generate predictions for vsr training
trainPredict1 = history1.model.predict(train)
testPredict1 = history1.model.predict(test)

# generate predictions for spr training
testPredict1 = historyp1.model.predict(testPP)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[Lv:len(trainPredict)+Lv, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(Lv*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(dataset[:,4])
plt.plot(trainPredictPlot)
plt.figure(dpi=300)
plt.plot(testPredictPlot)
plt.plot(datasetF[:,0])

plt.plot(range(len(test[:,0])),test[:,0],range(len(testPredict1)),testPredict1[:,0])
#plt.plot(testD[:,0])

plt.savefig( r'\\home.ansatt.ntnu.no\hosseind\Desktop\destination_path.png', format='png', dpi=1000)
plt.show()

savetxt('data2.csv', testPredict, delimiter=',')


#%%
# Creating Faulty signal
BiasF_tr , Label_tr = BiasFault(3,220,10,train_size)
(len(Label_tr)-np.count_nonzero(Label_tr[:,0]))/len(Label_tr)
BiasF_tst , Label_tst = BiasFault(3,3,4,test_size)
(len(Label_tst)-np.count_nonzero(Label_tst[:,0]))/len(Label_tst)

# Drift Fault
BiasF_tr , Label_tr = DriftFault(3,140,10,train_size)
(len(Label_tr)-np.count_nonzero(Label_tr[:,0]))/len(Label_tr)
BiasF_tst , Label_tst = DriftFault(3,25,4,test_size)
(len(Label_tst)-np.count_nonzero(Label_tst[:,0]))/len(Label_tst)


# for comparison with RUN.py
test = np.append(test, test, axis=0)
test_size = len(test)
# Drift Fault
BiasF_tr , Label_tr = DriftFaultPM(15,10,1,sn,8,2,train_size)
(len(Label_tr)-np.count_nonzero(Label_tr[:,0]))/len(Label_tr)
BiasF_tst , Label_tst = DriftFaultPM(1,1,0,sn,8,2,test_size)
(len(Label_tst)-np.count_nonzero(Label_tst[:,0]))/len(Label_tst)

# Bias Fault
BiasF_tr , Label_tr = BiasFaultPM(15,10,1,sn,8,2,train_size)
(len(Label_tr)-np.count_nonzero(Label_tr[:,0]))/len(Label_tr)
BiasF_tst , Label_tst = BiasFaultPM(1,1,0,sn,8,2,test_size)
(len(Label_tst)-np.count_nonzero(Label_tst[:,0]))/len(Label_tst)

train_f = np.add(train, BiasF_tr) 
test_f = np.add(test, BiasF_tst) 

# reshape dataset (injecting Lv delays)
trainDF = create_dataset(train_f, Lv)
testDF = create_dataset(test_f, Lv)
trainDF = trainDF.transpose()
testDF = testDF.transpose()

testPredict = history11.model.predict(testDF)

# residual calculations
res_tst = abs(np.subtract(testPredict,testDF))

res_tst2 = np.zeros((len(res_tst[:,0]),sn), dtype=d_type)
for i in range(sn):
    res_tst2[:,i] = res_tst[:,i*(Lv+1)]
del res_tst
res_tst = copy.deepcopy(res_tst2)
del res_tst2    

Predict_rnn = res_tst
Label_tst_f = Label_tst[Lv:,:]

plt.plot(range(len(Label_tst_f[:,1])),Label_tst_f[:,2],range(len(res_tst[:,0])),res_tst[:,1])


# Also including probability of identification
PD = []
PF = []
PI = []
eps, ep10 = -0.01, 0
while eps < 1:
    eps = eps + 0.001
    if eps > ep10:
        print(eps)
        ep10 = ep10 + 0.1
    mse01 = np.where(Predict_rnn<eps, 0, 1) 
    #False alarm and detection
    Pf, Pd, Pi = 0, 0, 0
    for k in range(len(Predict_rnn)):
        if (Label_tst_f[k,0] > 0.5):
            if sum(mse01[k,:])>0.5:
                Pf = Pf + 1
        else:
            if sum(mse01[k,:])>0.5:
                Pd = Pd + 1   
                l1 , l2 = 0, 0
                for j in range(sn):
                    if Label_tst_f[k,j+1] > 0.5:
                        l1 = l1 + 1
                        if mse01[k,j]>0.5:
                            l2 = l2 + 1
                if l1 == l2:
                    Pi = Pi + 1
    Pf, Pd = Pf/sum(Label_tst_f[:,0]), Pd/(len(Predict_rnn)-sum(Label_tst_f[:,0]))
    Pi =  Pi/(len(Predict_rnn)-sum(Label_tst_f[:,0]))
    PF.append(Pf), PD.append(Pd), PI.append(Pi)
PD = np.array(PD)
PF = np.array(PF)
PI = np.array(PI)

plt.plot(PF,PD,label="Probability of Detection")
plt.plot(PF,PI,label="Probability of Identification")
plt.xscale('log',base=10) 
plt.legend()
plt.show()

savetxt(r'/Users/hosseind/Desktop/data/PD_AE_WT_B.csv', PD, delimiter=',')
savetxt(r'/Users/hosseind/Desktop/data/PF_AE_WT_B.csv', PF, delimiter=',')
savetxt(r'/Users/hosseind/Desktop/data/PI_AE_WT_B.csv', PI, delimiter=',')


#%% AE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

testDF = np.zeros((sn*test_size, (sn+rn)), dtype=d_type)
testDF_ff = np.zeros((sn*test_size, sn), dtype=d_type)
test_ap = np.zeros((test_size,sn+rn,sn), dtype=d_type)
for i in range(sn):
    test_ap[:,:sn,i] = np.add(test[:,:sn], BiasF_tst[i*test_size:(i+1)*test_size, :]) 
    test_ap[:,sn:sn+rn,i] = copy.deepcopy(test[:,sn:sn+rn])
    testDF[i*(len(test_ap[:,0,0])):(i+1)*len(test_ap[:,0,0]),:] = copy.deepcopy(test_ap[:,:,i])
    testDF_ff[i*(len(test_ap[:,0,0])):(i+1)*len(test_ap[:,0,0]),:] = copy.deepcopy(test[:,:sn])
    
testDF_save = copy.deepcopy(testDF)

testPredict = history1.model.predict(testDF)

plt.plot(range(len(testDF_ff[:,0])),testDF_ff[:,0],range(len(testPredict[:,0])),testPredict[:,0])
plt.plot(range(len(Label_tst[:,1])),Label_tst[:,1])
# residual calculations
res_tst = abs(np.subtract(testPredict,testDF))

plt.plot(range(len(Label_tst[:,1])),Label_tst[:,2],range(len(res_tst[:,0])),res_tst[:,1])


# AE attack %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

trainDF = np.zeros(((sn+rn)*(Lv+1), sn*(train_size-np.maximum(Lv,Lp))), dtype=d_type)
trainDF_h = np.zeros(((sn+rn)*(Lv+1), sn*(train_size-np.maximum(Lv,Lp))), dtype=d_type)
train_ap = np.zeros((train_size,sn+rn,sn+1), dtype=d_type)
train_ap[:,:,0] = train
train_ap_h = np.zeros((train_size,sn+rn,sn+1), dtype=d_type)
train_ap_h[:,:,0] = train
testDF = np.zeros(((sn+rn)*(Lv+1), sn*test_size), dtype=d_type)
test_ap = np.zeros((test_size+Lv,sn+rn,sn), dtype=d_type)
testDF_h = np.zeros(((sn+rn)*(Lv+1), sn*test_size), dtype=d_type)
test_ap_h = np.zeros((test_size+Lv,sn+rn,sn), dtype=d_type)
for i in range(sn):
    train_ap[:,:sn,i+1] = np.add(train[:,:sn], BiasF_tr[i*train_size:(i+1)*train_size, :])
    train_ap[:,sn:sn+rn,i+1] = train[:,sn:sn+rn]
    
    train_ap_h[:,:sn,i+1] = train[:,:sn]
    train_ap_h[:,sn:sn+rn,i+1] = train[:,sn:sn+rn]

    test_ap[:,:sn,i] = np.add(np.append(train[train_size-Lv:train_size,:sn], test[:,:sn], axis=0), np.append(BiasF_tr[((i+1)*train_size)-Lv:((i+1)*train_size), :], BiasF_tst[i*test_size:(i+1)*test_size, :], axis=0))
    test_ap[:,sn:sn+rn,i] = np.append(train[train_size-Lv:train_size,sn:sn+rn], test[:,sn:sn+rn], axis=0)

    test_ap_h[:,:sn,i] = np.append(train[train_size-Lv:train_size,:sn], test[:,:sn], axis=0)
    test_ap_h[:,sn:sn+rn,i] = np.append(train[train_size-Lv:train_size,sn:sn+rn], test[:,sn:sn+rn], axis=0)
    
    Label_tr = np.delete(Label_tr, slice(i*(train_size-np.maximum(Lv,Lp)),i*(train_size-np.maximum(Lv,Lp))+np.maximum(Lv,Lp)), 0)
    # reshape dataset (injecting Lv delays)
    if (Lp-Lv)>0:
        traind = create_dataset(train_ap[abs(Lv-Lp):,:,i+1], Lv)
        traind_h = create_dataset(train_ap_h[abs(Lv-Lp):,:,i+1], Lv)
        testd = create_dataset(test_ap[:,:,i], Lv)
        testd_h = create_dataset(test_ap_h[:,:,i], Lv)
    else:
        traind = create_dataset(train_ap[:,:,i+1], Lv)
        traind_h = create_dataset(train_ap_h[:,:,i+1], Lv)
        testd = create_dataset(test_ap[:,:,i], Lv)
        testd_h = create_dataset(test_ap_h[:,:,i], Lv)
    trainDF[:,i*(len(traind[0,:])):(i+1)*len(traind[0,:])] = traind
    trainDF_h[:,i*(len(traind_h[0,:])):(i+1)*len(traind_h[0,:])] = traind_h
    testDF[:,i*(len(testd[0,:])):(i+1)*len(testd[0,:])] = testd
    testDF_h[:,i*(len(testd_h[0,:])):(i+1)*len(testd_h[0,:])] = testd_h
    
if (Lp-Lv)>0:
    traind = create_dataset(train_ap[abs(Lv-Lp):,:,0], Lv)
    traind_h = create_dataset(train_ap_h[abs(Lv-Lp):,:,0], Lv)
else:
    traind = create_dataset(train_ap[:,:,0], Lv)
    traind_h = create_dataset(train_ap_h[:,:,0], Lv)
    
trainDF = np.append(traind, trainDF, axis=1)
trainDF_h = np.append(traind_h, trainDF_h, axis=1)
trainDF = trainDF.transpose()
trainDF_h = trainDF_h.transpose()
testDF = testDF.transpose()
testDF_h = testDF_h.transpose()
del traind
del traind_h
del testd
del testd_h

testDF_save = copy.deepcopy(testDF)

testPredict = history11.model.predict(testDF)

#plt.plot(range(len(testDF_ff[:,0])),testDF_ff[:,0],range(len(testPredict[:,0])),testPredict[:,0])
#plt.plot(range(len(Label_tst[:,1])),Label_tst[:,1])

# residual calculations
res_tst = pow(np.subtract(testPredict,testDF),2)

#plt.plot(range(len(Label_tst[:,1])),Label_tst[:,2],range(len(res_tst[:,0])),res_tst[:,1])

res_tst2 = np.zeros((len(res_tst[:,0]),sn), dtype=d_type)
res_tst2[:,0] = res_tst[:,0]
res_tst2[:,1] = res_tst[:,1*(Lv+1)]
res_tst2[:,2] = res_tst[:,2*(Lv+1)]
res_tst2[:,3] = res_tst[:,3*(Lv+1)]
res_tst2[:,4] = res_tst[:,4*(Lv+1)]
del res_tst
res_tst = copy.deepcopy(res_tst2)
del res_tst2




# for comparison with RUN.py
res_tst2 = np.zeros((len(res_tst[:,0]),sn), dtype=d_type)
for i in range(sn):
    res_tst2[:,i] = res_tst[:,i*(Lv+1)]
del res_tst
res_tst = copy.deepcopy(res_tst2)
del res_tst2    

Predict_rnn = res_tst
Label_tst_f = Label_tst

i=90
plt.plot(range(len(Label_tst_f[Lv:-1,i+1])),Label_tst_f[Lv:-1,i+1],label="Label")
plt.plot(range(len(Predict_rnn[:,i])),Predict_rnn[:,i], label="Predict_RNN")
plt.legend()
plt.show()

plt.plot(range(len(testPredict[:,i])),testPredict[:,i],label="Label")
plt.plot(range(len(testDF[:,i])),testDF[:,i], label="Predict_RNN")
plt.legend()
plt.show()

PD = []
PF = []
eps = -0.01
while eps < 1:
    eps = eps + 0.001
    mse01 = np.where(Predict_rnn<eps, 0, 1) 
    #False alarm and detection
    Pf, Pd = 0, 0
    for k in range(len(Predict_rnn)):
        if (Label_tst_f[k+Lv,0] > 0.5):
            if sum(mse01[k,:])>0.5:
                Pf = Pf + 1
        else:
            if sum(mse01[k,:])>0.5:
                Pd = Pd + 1                
    Pf, Pd = Pf/sum(Label_tst_f[Lv:-1,0]), Pd/(len(Predict_rnn)-sum(Label_tst_f[Lv:-1,0]))
    PF.append(Pf), PD.append(Pd)
PD = np.array(PD)
PF = np.array(PF)

plt.plot(PF,PD,label="Probability of Detection")
plt.xscale('log',base=10) 
plt.legend()
plt.show()




# DAE
Label = np.zeros((train_size-np.maximum(Lv,Lp),sn+1),dtype='int8')
Label_tr = np.append(Label,Label_tr, axis=0)

with tf.device('/CPU:0'):
    history22 = vs2(trainDF, trainDF_h)
    testPredict2 = history22.model.predict(testDF)


savetxt(r'/Users/hosseind/Desktop/data/datatst_Lc10_wdrift.csv', res_tst[:,:sn], delimiter=',')
#savetxt(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\labeltst_Lc10_wdrift.csv', Label_tstc, delimiter=',')
savetxt(r'/Users/hosseind/Desktop/data/labeltst_Lc10_wdrift.csv', Label_tst, delimiter=',')



keys={
    (0,0,0,0,0): 0,
    (0,0,0,0,1): 1,
    (0,0,0,1,0): 2,
    (0,0,0,1,1): 3,
    (0,0,1,0,0): 4,
    (0,0,1,0,1): 5,
    (0,0,1,1,0): 6,
    (0,0,1,1,1): 7,
    (0,1,0,0,0): 8,
    (0,1,0,0,1): 9,
    (0,1,0,1,0): 10,
    (0,1,0,1,1): 11,
    (0,1,1,0,0): 12,
    (0,1,1,0,1): 13,
    (0,1,1,1,0): 14,
    (0,1,1,1,1): 15,
    (1,0,0,0,0): 16,
    (1,0,0,0,1): 17,
    (1,0,0,1,0): 18,
    (1,0,0,1,1): 19,
    (1,0,1,0,0): 20,
    (1,0,1,0,1): 21,
    (1,0,1,1,0): 22,
    (1,0,1,1,1): 23,
    (1,1,0,0,0): 24,
    (1,1,0,0,1): 25,
    (1,1,0,1,0): 26,
    (1,1,0,1,1): 27,
    (1,1,1,0,0): 28,
    (1,1,1,0,1): 29,
    (1,1,1,1,0): 30,
    (1,1,1,1,1): 31}


Label_tst_ohe = np.zeros((len(Label_tst)), dtype=d_type)
for i in range(len(Label_tst)):
    Label_tst_ohe[i] = keys.get(tuple(Label_tst[i,1:]))
#Label_tst_ohe = keras.utils.to_categorical(Label_tst_ohe, num_classes=pow(2,sn), dtype='int8')


Tau = np.arange(-0.01, 0.039, 0.0001)
Tau = np.append(Tau, np.arange(0.04, 0.95, 0.005))
Tau = np.append(Tau, np.arange(0.951, 1.01, 0.0001))
con_int = np.zeros((len(Tau),pow(2,sn),pow(2,sn)), dtype=d_type)
con_sep = np.zeros((len(Tau),sn,2,2), dtype=d_type)
clsfr = np.zeros((len(Tau),len(Label_tst),sn), dtype=d_type)
clsfr_ohe = np.zeros((len(Tau),len(Label_tst)), dtype=d_type)
for k in range(len(Tau)):
    print(k)
    for i in range(len(Label_tst)):
        #if np.amax(testPredict[i,:]) > Tau[k]:
        for j in  np.where(res_tst[i, :sn] > Tau[k]):
            clsfr[k,i,j] = 1

        clsfr_ohe[k,i] = keys.get(tuple(clsfr[k,i,:]))
                
    #clsfr_ohe = keras.utils.to_categorical(clsfr_ohe, num_classes=pow(2,sn), dtype='int8')
    con_int[k,:,:] = confusion_matrix(y_pred=clsfr_ohe[k,:], y_true=Label_tst_ohe, labels=np.arange(0,pow(2,sn),1))
    con_sep[k,:,:,:] = multilabel_confusion_matrix(Label_tst[:,1:], clsfr[k,:])

scipy.io.savemat(r'/Users/hosseind/Desktop/data/con_int.mat', dict(con_int=con_int))
scipy.io.savemat(r'/Users/hosseind/Desktop/data/con_sep.mat', dict(con_sep=con_sep))



testPredict_DAE0 = np.zeros((len(Label_tst),sn),dtype=d_type)
for i in range(len(Label_tst)):
    
    for j in  np.where(res_tst[i, :sn] > 0.04):
        for k in j:
            testPredict_DAE0[i,k] = pow(np.subtract(testPredict2[i,k*(Lv+1)],testDF_h[i,k*(Lv+1)]),2)
    for j in  np.where(Label_tst[i, 1:] > 0.5):
        for k in j:
            if res_tst[i, k] < 0.0262:
                testPredict_DAE0[i,k] = pow(np.subtract(testDF_save[i,k*(Lv+1)],testDF_h[i,k*(Lv+1)]),2)
savetxt(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\MSE.csv', testPredict_DAE0, delimiter=',')


testPredict_DAE1 = np.zeros((len(Label_tst),sn),dtype=d_type)
for i in range(len(Label_tst)):
    for j in  np.where(res_tst[i, :sn] > 0.065):
        for k in j:
            if Label_tst[i,k+1] > 0.5:
                testPredict_DAE1[i,k] = pow(np.subtract(testPredict2[i,k*(Lv+1)],testDF_h[i,k*(Lv+1)]),2)
savetxt(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\MSE_cls.csv', testPredict_DAE1, delimiter=',')

testPredict_DAE2 = np.zeros((len(Label_tst),sn),dtype=d_type)
for i in range(len(Label_tst)):
    
    for j in  np.where(Label_tst[i, 1:] > 0.5):
        for k in j:
            if res_tst[i, k] > 0.07:
                testPredict_DAE2[i,k] = (np.subtract(testPredict2[i,k*(Lv+1)],testDF_h[i,k*(Lv+1)]))
            else:
                testPredict_DAE2[i,k] = (np.subtract(testDF_save[i,k*(Lv+1)],testDF_h[i,k*(Lv+1)]))
savetxt(r'/Users/hosseind/Desktop/data/MSE_label.csv', testPredict_DAE2, delimiter=',')

mse_cls = np.zeros((sn),dtype=d_type)
mse_label = np.zeros((sn),dtype=d_type)
for i in range(sn):
    idx1 = np.nonzero(testPredict_DAE1[:,i])
    idx2 = np.nonzero(testPredict_DAE2[:,i])
    mse_cls[i] = np.mean(testPredict_DAE1[idx1,i])
    mse_label[i] = np.mean(testPredict_DAE2[idx2,i])
mse_cls_f = np.mean(mse_cls)
mse_label_f = np.mean(mse_label)




for i in range(sn):
    idx = np.nonzero(testPredict_DAE[:,i])
    mse = np.mean(testPredict_DAE[idx,i])




plt.plot(testDF[:,22])
plt.plot(Label_tst[:,1])
plt.plot(Label_tst[:,2])
plt.plot(Label_tst[:,3])
plt.plot(Label_tst[:,4])
plt.plot(Label_tst[:,5])
plt.plot(test[:,0])
plt.plot(testPredict[:,1])


trainPredict = historyC.model.predict(resf_tr)
trainPredict = np.concatenate(trainPredict, axis=1) # only for multi_output classifier
savetxt(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\datatr_Lc10_wdrift.csv', trainPredict, delimiter=',')
savetxt(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\labeltr_Lc10_wdrift.csv', Label_tr, delimiter=',')



#%% saving models
import pickle
# save the model to disk
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\history1.sav'
pickle.dump(history1, open(filename, 'wb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\history2.sav'
pickle.dump(history2, open(filename, 'wb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\history3.sav'
pickle.dump(history3, open(filename, 'wb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\history4.sav'
pickle.dump(history4, open(filename, 'wb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\history5.sav'
pickle.dump(history5, open(filename, 'wb'))
 
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp1.sav'
pickle.dump(historyp1, open(filename, 'wb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp2.sav'
pickle.dump(historyp2, open(filename, 'wb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp3.sav'
pickle.dump(historyp3, open(filename, 'wb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp4.sav'
pickle.dump(historyp4, open(filename, 'wb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp5.sav'
pickle.dump(historyp5, open(filename, 'wb'))

filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp.sav'
pickle.dump(historyp, open(filename, 'wb'))

 
# load the model from disk
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\history1.sav'
history1 = pickle.load(open(filename, 'rb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\history2.sav'
history2 = pickle.load(open(filename, 'rb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\history3.sav'
history3 = pickle.load(open(filename, 'rb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\history4.sav'
history4 = pickle.load(open(filename, 'rb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\history5.sav'
history5 = pickle.load(open(filename, 'rb'))

filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp1.sav'
historyp1 = pickle.load(open(filename, 'rb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp2.sav'
historyp2 = pickle.load(open(filename, 'rb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp3.sav'
historyp3 = pickle.load(open(filename, 'rb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp4.sav'
historyp4 = pickle.load(open(filename, 'rb'))
filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp5.sav'
historyp5 = pickle.load(open(filename, 'rb'))

filename = r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\historyp.sav'
historyp = pickle.load(open(filename, 'rb'))

#%% confusion matrices
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import scipy.io
#first approach
#keys={
#    (0,0,0,0): 0,
#    (1,0,0,0): 1,
#    (0,1,0,0): 2,
#    (0,0,1,0): 3,
#    (0,0,0,1): 4,
#    (1,1,0,0): 5,
#    (1,0,1,0): 6,
#    (1,0,0,1): 7,
#    (0,1,1,0): 8,
#    (0,1,0,1): 9,
#    (0,0,1,1): 10,
#    (1,1,1,0): 11,
#    (1,1,0,1): 12,
#    (1,0,1,1): 13,
#    (0,1,1,1): 14,
#    (1,1,1,1): 15}

"""keys={
    (0,0,0,0): 0,
    (0,0,0,1): 1,
    (0,0,1,0): 2,
    (0,0,1,1): 3,
    (0,1,0,0): 4,
    (0,1,0,1): 5,
    (0,1,1,0): 6,
    (0,1,1,1): 7,
    (1,0,0,0): 8,
    (1,0,0,1): 9,
    (1,0,1,0): 10,
    (1,0,1,1): 11,
    (1,1,0,0): 12,
    (1,1,0,1): 13,
    (1,1,1,0): 14,
    (1,1,1,1): 15}"""

keys={
    (0,0,0,0,0): 0,
    (0,0,0,0,1): 1,
    (0,0,0,1,0): 2,
    (0,0,0,1,1): 3,
    (0,0,1,0,0): 4,
    (0,0,1,0,1): 5,
    (0,0,1,1,0): 6,
    (0,0,1,1,1): 7,
    (0,1,0,0,0): 8,
    (0,1,0,0,1): 9,
    (0,1,0,1,0): 10,
    (0,1,0,1,1): 11,
    (0,1,1,0,0): 12,
    (0,1,1,0,1): 13,
    (0,1,1,1,0): 14,
    (0,1,1,1,1): 15,
    (1,0,0,0,0): 16,
    (1,0,0,0,1): 17,
    (1,0,0,1,0): 18,
    (1,0,0,1,1): 19,
    (1,0,1,0,0): 20,
    (1,0,1,0,1): 21,
    (1,0,1,1,0): 22,
    (1,0,1,1,1): 23,
    (1,1,0,0,0): 24,
    (1,1,0,0,1): 25,
    (1,1,0,1,0): 26,
    (1,1,0,1,1): 27,
    (1,1,1,0,0): 28,
    (1,1,1,0,1): 29,
    (1,1,1,1,0): 30,
    (1,1,1,1,1): 31}


Label_tst_ohe = np.zeros((len(Label_tst)), dtype=d_type)
for i in range(len(Label_tst)):
    Label_tst_ohe[i] = keys.get(tuple(Label_tst[i,1:]))
#Label_tst_ohe = keras.utils.to_categorical(Label_tst_ohe, num_classes=pow(2,sn), dtype='int8')


Tau = np.arange(-0.01, 0.039, 0.0001)
Tau = np.append(Tau, np.arange(0.04, 0.95, 0.005))
Tau = np.append(Tau, np.arange(0.951, 1.01, 0.0001))
con_int = np.zeros((len(Tau),pow(2,sn),pow(2,sn)), dtype=d_type)
con_sep = np.zeros((len(Tau),sn,2,2), dtype=d_type)
clsfr = np.zeros((len(Tau),len(Label_tst),sn), dtype=d_type)
clsfr_ohe = np.zeros((len(Tau),len(Label_tst)), dtype=d_type)
for k in range(len(Tau)):
    print(k)
    for i in range(len(Label_tst)):
        #if np.amax(testPredict[i,:]) > Tau[k]:
        for j in  np.where(testPredict[i,:] > Tau[k]):
            clsfr[k,i,j] = 1

        clsfr_ohe[k,i] = keys.get(tuple(clsfr[k,i,:]))
                
    #clsfr_ohe = keras.utils.to_categorical(clsfr_ohe, num_classes=pow(2,sn), dtype='int8')
    con_int[k,:,:] = confusion_matrix(y_pred=clsfr_ohe[k,:], y_true=Label_tst_ohe, labels=np.arange(0,pow(2,sn),1))
    con_sep[k,:,:,:] = multilabel_confusion_matrix(Label_tst[:,1:], clsfr[k,:])

scipy.io.savemat(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\con_int.mat', dict(con_int=con_int))
scipy.io.savemat(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\con_sep.mat', dict(con_sep=con_sep))

# import
#bdict = scipy.io.loadmat(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\con_sep.mat')
#bdict.get('con_sep')
#%%

#second approach
y_true = np.array([[0,0,1], [1,1,0],[0,1,0]])
y_pred = np.array([[0,0,1], [1,0,1],[1,0,0]])

labels = ["A", "B", "C"]

conf_mat_dict={}

for label_col in range(len(labels)):
    y_true_label = y_true[:, label_col]
    y_pred_label = y_pred[:, label_col]
    conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)

for label, matrix in conf_mat_dict.items():
    print("Confusion matrix for label {}:".format(label))
    print(matrix)

#or
from sklearn.metrics import multilabel_confusion_matrix

aa = multilabel_confusion_matrix(y_true, y_pred)

#%%
from mlxtend.plotting import plot_confusion_matrix
multiclass = np.array([[6203, 6, 9, 3, 3, 49],
                       [1, 90, 0, 0, 0, 0],
                       [0, 0, 78, 0, 1, 0],
                       [7, 0, 0, 103, 0, 0]
                       [1, 0, 0, 4, 86, 0]
                       [6, 0, 0, 0, 0, 90]])

fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True)
plt.show()


[0.031215353554822172, 0.016682209447026253, 0.017551349475979805, 0.040120724588632584, 0.03129797801375389, 0.05182155966758728, 0.0034100902266800404, 0.004779850598424673, 0.009092110209167004, 0.006520670838654041, 0.012706362642347813]

testappend = testPredict
testappend = np.append(testappend,testPredict, axis=0)

labeld=Label_tst
labeld = np.append(labeld,Label_tst, axis=0)

savetxt(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\datatst_Lc10_wdrift.csv', testappend, delimiter=',')
savetxt(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\labeltst_Lc10_wdrift.csv', labeld, delimiter=',')

trainPredict = historyC.model.predict(res_tr)
trainPredict = np.concatenate(trainPredict, axis=1) # only for multi_output classifier

testappend2 = trainPredict
testappend2 = np.append(testappend2,trainPredict, axis=0)

labeld2=Label_tr
labeld2 = np.append(labeld2,Label_tr, axis=0)


savetxt(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\datatr_Lc10_wdrift.csv', testappend2, delimiter=',')
savetxt(r'\\home.ansatt.ntnu.no\hosseind\Desktop\data\labeltr_Lc10_wdrift.csv', labeld2, delimiter=',')

#%%
# creates bias fault and label (for 'max01' normalizerr)
def BiasFaultPM(Nf,N,Nn,sn, m, x,size):
    # x = minimum space between two starting fault is x
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



