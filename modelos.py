# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:17:40 2021

@author: martuca
"""

import tensorflow as tf
from sklearn.model_selection import cross_validate
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def models_comparation(X,Y):
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
    rf_gs = GridSearchCV(rf, params_rf, cv=5)
    #fit model to training data
    rf_gs.fit(X_t, Y_t)
    #save best model
    rf_best = rf_gs.best_estimator_
    #check best n_estimators value
    print(rf_gs.best_params_)
    
    
    # Creación del modelo de LSTM (NO FUNCIONA PORQUE ME OBLIGA A PONER EL PARÁMETRO UNITS)
    #create a dictionary of all values we want to test for n_estimators
    units = [30, 50, 70, 90]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid']
    batch_size = [8, 16, 32, 48]
    epochs = [50, 100, 150, 200, 250]
    optimizer = ['Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    #params_LSTM = dict(units=units, dropout_rate=dropout_rate, activation=activation,
                     # batch_size=batch_size, optimizer=optimizer, epochs=epochs)
    params_LSTM = dict(dropout_rate=dropout_rate, activation=activation,
                      batch_size=batch_size, optimizer=optimizer, epochs=epochs)
    #use gridsearch to test all values for n_estimators
    LSTM_gs = GridSearchCV(tf.keras.layers.LSTM(30), params_LSTM, cv=5)
    #fit model to training data
    LSTM_gs.fit(X_t, Y_t)
    #save best model
    LSTM_best = LSTM_gs.best_estimator_
    #check best n_estimators value
    print(LSTM_gs.best_params_)
    
    
    # Modelos a analizar
    models = []
    models.append(('RF', rf_best()))
    models.append('LSTM', LSTM_best())
    
    
    # 10-fold cross validation pa estimar la precisión (dividimos los datos en 10 partes, 9 para entrenamiento y 1 para validación)
    num_folds = 10
    num_instances = len(X_t)
    seed = 7
    # Usamos la precisión como métrica para evaluar los modelos (aciertos/total)
    scoring = 'accuracy'
    
    # Evaluación de cada modelo
    results = []
    names = []
    print("Scores for each algorithm:")
    for name, model in models:
        kfold = cross_validate.KFold(n = num_instances, n_folds = num_folds, random_state = seed)
        cv_results = cross_validate.cross_val_score(model, X_t, Y_t, cv = kfold, scoring = scoring)
        results.append(cv_results)
        names.append(name)
        model.fit(X_t, Y_t)
        predictions = model.predict(X_val)
        print(name, accuracy_score(Y_val, predictions)*100)
        print(matthews_corrcoef(Y_val, predictions))
        print()