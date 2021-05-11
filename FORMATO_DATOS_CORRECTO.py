# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:38:17 2021

@author: martuca
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import os

#CREACIÓN DEL FORMATO CORRECTO DE DATOS PARA METER EN STMS

#Obtenemos un único dataframe con todos los datos cuyos índices son la etiqueta y tenemos los datos de cada columna en cada fila.

# Búsqueda de archivos por las carpetas
directorio_origen = 'C:/Users/martuca/Documents/Master_Bioinformatica/TFM/Algoritmo/Data'
archivo_prueba = 'C:/Users/martuca/Documents/Master_Bioinformatica/TFM/Algoritmo/Data/M_J/M_J1613658451212/M_J1613658451212_TOBII_output.tsv'
X_train = []
X_test = []
series_test = 1
series_train = 1

os.chdir(directorio_origen)
for root, dirs, files in os.walk (".", topdown = False):
    for name in files:
        if ".tsv" in name:
            X=[]
            cols = ['Label', 'GazePointXLeft', 'GazePointYLeft', 'GazePointXRight', 'GazePointYRight', 'GazePointX', 'GazePointY', 'PupilSizeLeft', 'PupilSizeRight']
            dataset = pd.read_csv(root + "/" + name,  sep='\t',  usecols = cols, skiprows=17)
            dataset_2 = dataset[dataset.GazePointXLeft != -1]
            
            # Dividimos cada file de cada paciente en 4 para aumentar los datos
            dataset_3 = np.array_split(dataset_2, 4)
            
            for i in dataset_3:
                i.insert(1, 'Time_index', range(0,len(i)), True)
                i_2 = i[i.GazePointXLeft != 'nan']
                i_2 = i_2[['Time_index', 'Label', 'GazePointXLeft', 'GazePointYLeft', 'GazePointXRight', 'GazePointYRight', 'GazePointX', 'GazePointY', 'PupilSizeLeft', 'PupilSizeRight']]
                X.append(i_2)
                
                
            #Para separar de cada paciente el 25% de los datos para test y el 75% para train en caso de querer hacerlo aquí y no despues
            x_test = random.choice(X)
            X_test.append(x_test)
            for i in X:
                if i.equals(x_test):
                    pass
                else:
                    X_train.append(i)

for i in X_test:
    i.insert(0, 'Series', list([series_test]*len(i)), True)
    series_test += 1

for i in X_train:
    i.insert(0, 'Series', list([series_train]*len(i)), True)
    series_train += 1

X_t = pd.concat(X_train)
X_t2 = X_t.dropna()
X_t2.to_csv('C:/Users/martuca/Documents/Master_Bioinformatica/TFM/Algoritmo/LSTM/Propio/X_t.csv', index=False)

X_val = pd.concat(X_test)
X_val2 = X_val.dropna()
X_val2.to_csv('C:/Users/martuca/Documents/Master_Bioinformatica/TFM/Algoritmo/LSTM/Propio/X_val.csv', index=False)
