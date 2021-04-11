# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:31:15 2021

@author: martuca
"""

import numpy as np
import tensorflow as tf
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

Y = np.array([1,0,0,1])

#EL PROBLEMA ES QUE LOS DATOS SON TRIDIMENSIONALES Y NO SE PUEDE HACER FIT ASI PARA RF

# División de los archivos en train (2/3) y test (1/3) aproximadamente
test_percentage = 0.3
# Randomerización de que parte es test y que parte es train
seed = 7
X_t, X_val, Y_t, Y_val = train_test_split(X, Y, test_size = test_percentage, random_state = seed)

    
# Creación del modelo de Random Forest
rf = RandomForestClassifier()

#Creación del diccionario con los valores para cada parámetro que queremos testar
params_rf = {'n_estimators': [50, 100, 200]}
#Use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=2) #CAMBIAR A 10! LO PUSE ASÍ PARA TRABAJAR CON DATOS SIMPLES!!!
#fit model to training data
rf_gs.fit(X_t, Y_t)
#save best model
rf_best = rf_gs.best_estimator_
#check best n_estimators value
print(rf_gs.best_params_)


# Creation of a multivariate time series LSTM model (NO FUNCIONA PORQUE ME OBLIGA A PONER EL PARÁMETRO UNITS)
#create a dictionary of all values we want to test for n_estimators
units = [30, 50, 70, 90]
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
    scores_lstm.append(sqrt(mean_squared_error(Y_t, Y_pred)))

max_score_lstm = max(scores_lstm)
LSTM_best_config_index = scores_lstm.index(max_score_lstm)
LSTM_best_config = params_LSTM[LSTM_best_config_index]
units, dropout_rate, activation, batch_size, epochs, optimizer = LSTM_best_config
LSTM = tf.keras.layers.LSTM(units=units, activation=activation, dropout=dropout_rate)
print(LSTM)
LSTM_best = LSTM.fit(X_t, Y_t, epochs=epochs, batch_size=batch_size, verbose=0)

# Modelos a analizar
models = []
models.append(('RF', rf_best()))
models.append('LSTM', LSTM_best())