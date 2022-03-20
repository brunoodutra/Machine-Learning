# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:21:05 2019

@author: bruno
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt 
 
xi       = -5.0
xf       = 5.0
numDados = 100
semente  = 4727
 
#iniciando o gerador
conjAleaNum = np.random.RandomState(seed=semente)
 
#gera os número
Xi = conjAleaNum.uniform(low=xi, high=xf, size=numDados)
 
#cria numDados entre 0 e numDados igualmente espaçados
x  =np.linspace(1, 2, numDados)

a=4; b=2; c=4;
y=(a*x**2 +b*x +c) ; 

data=np.zeros((np.size(y),4))
k=3

for k in range(np.size(y)):
    data[k]=[y[k-1] ,y[k-2], y[k-3] ,y[k-4]] # modelo autorregressivo  rede NAR

data[0]=[10,10,10,10]
data[2]=[y[1],10,10,10]
data[3]=[y[2],y[1],10,10]
data[4]=[y[3],y[2],y[1],10]

#data=x**2 +x;
#x = data = np.linspace(1,2,200)
#y = x*4 + np.random.randn(*x.shape) * 0.3

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(1, activation='linear'))


model.compile(optimizer='adam', loss='mse', metrics=['mse'])

weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init)) 


model.fit(data,y, batch_size=1, epochs=100, shuffle=True, verbose=1)

weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))

predict = model.predict(data)

plt.plot(x, predict, 'b', x , y, 'k.')
plt.show()