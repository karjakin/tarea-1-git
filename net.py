# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:18:13 2022

@author: jairc
se implemento la red neuronal con 784 datos de entrada, corespondientes a los pixeles de los
numeros, tiene un capa intermedia de 30 neuronas y una capa de salida de 10
se utilizo el algoritmo de stochastic gradient descent, para optimizar la red
con una taza de aprendizaje de 3 con 30 generaciones.
La red neuronal ha mejorado respecto a la primera epoca, empezo con un porcentaje del 90%,
para la epoca 10 llego a un porcentaje del 94.26 %
para la epoca 20 mejoro apenas a 94.87 %
para la epoca 23 se tiene una mejora del 95.14 %, despues de empezo a bajar 
finalmente se conseguio para la epoca 29, una mejora del 94.87 %
"""
import mnist_loader
import network
import pickle
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
net=network.Network([784,30,10])
net.SGD( training_data, 30, 10, 4.0, 0.2, test_data=test_data)
archivo = open("red_prueba1.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()
#leer el archivo
archivo_lectura = open("red_prueba.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()
net.SGD( training_data, 10, 50, 0.5, 0.1, test_data=test_data)
archivo = open("red_prueba.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()

"""

Para una red con cross entropy
para la primera epoca, empezo con un porcentaje del 81.56%,
para la epoca 10 llego a un porcentaje del 94.30 %
para la epoca 20 a 94.96 %
para la epoca 23 se tiene una mejora del 94.78 % 
finalmente se conseguio para la epoca 29, una mejora del 94.80 %
curiosamente bajo su rendimiento respecto al primer intento, sin cross entropy
"""

"""
implementando Stochastic Gradient Descent with Momentum sin cross entropy
se logro mejorar el rendimiento de la red
para la primera epoca, empezo con un porcentaje del 84.10%,
para la epoca 10 llego a un porcentaje del 94.67 %
para la epoca 20 mejoro a 95.22 %
finalmente se conseguio para la epoca 29, una mejora del 95.46 %
"""
"""
implementando Stochastic Gradient Descent with Momentum sin cross entropy
con una tasa de aprenidizaje de 4 y una fricción de 0.2
para la primera epoca, empezo con un porcentaje del 90.12 %,
para la epoca 10 llego a un porcentaje del 94.71%
para la epoca 20 mejoro a 95.03 %
para la epoca 27 se tiene una mejora del 95.06 %, mejora maxima
finalmente se conseguio para la epoca 29, una mejora del 94.62 %

"""