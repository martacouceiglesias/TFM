# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:18:09 2021

@author: marta
"""

import numpy as np
import pandas as pd
import os
import modelos

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
            #dataset_2 = dataset[dataset.GazePointXLeft != -1]
            # Dividimos cada file de cada paciente en 4 para aumentar los datos
            #dataset_3 = np.array_split(dataset_2, 4)
            dataset_3 = np.array_split(dataset, 4)
            X.extend(dataset_3[:])

# Siendo EN = enfermedad neurodegenerativa y H = healthy
labels = ['EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H','EN','H']

#De esta forma tenemos una lista igual que X para Y. Habrá que sustituir los datos
#de todos los dataframes por las Y correspondientes para tener una lista con dataframes
#igual que X y así poder usar el fit en los modelos.
Y = []

for i in X:
    Y.append(i.copy())

for i in range(0,len(Y)):
    Y[i][Y[i]<5000000]=labels[i]


modelos.models_comparation(X, Y)