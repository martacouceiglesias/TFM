# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:31:15 2021

@author: martuca
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


# Búsqueda de archivos por las carpetas
directorio_origen = 'C:/Users/martuca/Documents/Master_Bioinformatica/TFM/Algoritmo/Data'
X = []
#Y = []

os.chdir(directorio_origen)
for root, dirs, files in os.walk (".", topdown = False):
    for name in files:
        if ".tsv" in name:
            cols = ['TimeStamp','GazePointXLeft', 'GazePointYLeft', 'GazePointXRight', 'GazePointYRight', 'GazePointX', 'GazePointY', 'PupilSizeLeft', 'PupilSizeRight']
            dataset = pd.read_csv(root + "/" + name,  sep='\t',  usecols = cols, skiprows=17)
            #CREO QUE NO SE PUEDEN ELIMINAR LOS -1 PORQUE SI NO LOS DATAFRAMES SON DE DISTINTAS DIMENSIONES Y LUEGO EN EL FIT NO DA FORMADO UN ARRAY 2D
            dataset_2 = dataset[dataset.GazePointXLeft != -1]
            # Dividimos cada file de cada paciente en 4 para aumentar los datos
            dataset_3 = np.array_split(dataset_2, 4)
            #dataset_3 = np.array_split(dataset, 4) EN CASO DE QUE NO SE PUEDAN ELIMINAR LOS -1
            X.extend(dataset_3[:])

# Siendo EN = enfermedad neurodegenerativa y H = healthy
Y = ['EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H']


'''
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

Y = np.array([1,0,0,1])
'''

# División de los archivos en train (2/3) y test (1/3) aproximadamente
test_percentage = 0.3
# Randomerización de que parte es test y que parte es train
seed = 7
X_t, X_val, Y_t, Y_val = train_test_split(X, Y, test_size = test_percentage, random_state = seed)

    

# Creation of a multivariate time series LSTM model (NO FUNCIONA PORQUE ME OBLIGA A PONER EL PARÁMETRO UNITS)
#create a dictionary of all values we want to test for n_estimators
'''units = [30, 50, 70, 90]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid']
batch_size = [8, 16, 32, 48]
epochs = [50, 100, 150, 200, 250]
optimizer = ['Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

params_LSTM = []
X_t = X_t.reshape(X_t.shape[0],1, X_t.shape[1])
X_t = tf.cast(X_t, tf.float32)
X_val = X_val.reshape(X_val.shape[0],1, X_val.shape[1])
X_val = tf.cast(X_val, tf.float32)

for i in units:
    for j in dropout_rate:
        for k in activation:
            for l in batch_size:
                for m in epochs:
                    for n in optimizer:
                        params = [i,j,k,l,m,n]
                        params_LSTM.append(params)
scores_lstm = []
for i in params_LSTM:
    units, dropout_rate, activation, batch_size, epochs, optimizer = i
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.fit(X_t, Y_t, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=2, shuffle=False)
    Y_pred = model.predict(X_val, verbose=0)
    scores_lstm.append(sqrt(mean_squared_error(Y_t, Y_pred)))'''

'''
X_t = X_t.reshape(X_t.shape[0],1, X_t.shape[1])
X_t = tf.cast(X_t, tf.float32)
X_val = X_val.reshape(X_val.shape[0],1, X_val.shape[1])
X_val = tf.cast(X_val, tf.float32)
'''

units = 30
dropout_rate = 0.1
activation = 'sigmoid'
batch_size = 16
epochs = 100
optimizer = 'Adamax'

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units, activation=activation, dropout=dropout_rate))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer=optimizer)
model.fit(X_t, Y_t, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=2, shuffle=False)
Y_pred = model.predict(X_val, verbose=0)
scores_lstm = sqrt(mean_squared_error(Y_t, Y_pred))

print(1-scores_lstm)


