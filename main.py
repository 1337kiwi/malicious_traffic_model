# Python Program using tensorflow to differentiate malicious and benign SSH and FTP traffic

import pandas
import random
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import keras.metrics as km
from tensorflow import keras


dataframe = pandas.read_csv('test.csv', engine='python')

del dataframe['Dst Port']
del dataframe['Protocol']
del dataframe['Timestamp']

features = dataframe.values
x = features[:, :-1]
y = features[:, -1]

scaler = MinMaxScaler() # MinMaxScaler()
scaler.fit(x) # fit the data
x_normalized = scaler.transform(x) # Normalize the data

samples = {}
labels_dict = {}

for i in range(len(y)):
    label = y[i]
    sample = x[i]

    if label not in samples:
        samples[label] = []
        labels_dict[label] = []
    
    samples[label].append(sample)
    if label == 'Benign':
        labels_dict[label].append(0)
    else:
        labels_dict[label].append(1)

for key, value in samples.items():
    if key != 'Benign': 
        num_samples = len(value)
        for i in range(num_samples):
            index = random.randint(0, len(samples['Benign']) - 1)
            benign_sample = samples['Benign'][index]
            samples[key].append(benign_sample)
            labels_dict[key].append(0)

ml_models = {}

for key, value in samples.items():
    if key != 'Benign':
        ml_models[key] = Sequential()
        # Dense(38) is the best performing model for the data. It has 38 neurons in the first layer. 
        ml_models[key].add(Dense(38, input_dim=76, activation='relu')) # Determined through testing and experimentation. 
        ml_models[key].add(Dense(1, activation='sigmoid')) # Sigmoid activation function is the best performing for the data.
        ml_models[key].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', km.TruePositives(), km.FalseNegatives(), km.TrueNegatives(), km.FalsePositives()])

        ml_models[key].fit(np.array(samples[key]), np.array(labels_dict[key]), 100, 100, verbose=1)