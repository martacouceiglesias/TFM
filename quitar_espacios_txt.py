# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:32:39 2021

@author: martuca
"""

         
inputFile = open('C:/Users/martuca/Documents/Master_Bioinformatica/TFM/Algoritmo/LSTM/Propio/X_t.txt', 'r') 
exportFile = open('X_t_espacios.txt', 'w')
for line in inputFile:
   new_line = line.replace('\t', ' ')
   if len(new_line.split(' ')) == 11:
       exportFile.write(new_line) 
   else:
       pass

inputFile.close()
exportFile.close()

inputFile = open('C:/Users/martuca/Documents/Master_Bioinformatica/TFM/Algoritmo/LSTM/Propio/X_val.txt', 'r') 
exportFile = open('X_val_espacios.txt', 'w')
for line in inputFile:
   new_line = line.replace('\t', ' ')
   if len(new_line.split(' ')) == 11:
       exportFile.write(new_line) 
   else:
       pass


inputFile.close()
exportFile.close()
