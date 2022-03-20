# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:21:05 2019

@author: bruno
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import functions
from keras.optimizers import SGD
from keras.models import Model
import tensorflow as tf
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt 


ref_arquivo = open("Registro170620191505.txt","r")

i=0; nit=0;
for j in ref_arquivo:
    if j!= '\n':
        nit=nit+1
ref_arquivo.close()

#cria numDados entre 0 e numDados igualmente espaçados
Ts=0.015
t  =np.linspace(1, Ts*nit-Ts, nit)
    
ref_arquivo = open("Registro170620191505.txt","r")    
u1=np.zeros((nit,1))
u2=np.zeros((nit,1))
u3=np.zeros((nit,1))
y=np.zeros((nit,1))

for linha in ref_arquivo:
    if linha!= '\n':
        l= linha.split(',')
        u1[i]=[float(l[0])]
        u2[i]=[float(l[1])]
        u3[i]=[float(l[2])]
        y[i]=[float(l[3])]
        
        #print('valores', y[i])
        i=i+1;
        
u2=functions.Norm(u2,1,0)
u3=functions.Norm(u3,1,0)
y=functions.Norm(y,1,0)




Win=30; incre=15;
u2=functions.Mov(u2,Win,incre)
u3=functions.Mov(u3,Win,incre)
y=functions.Mav(y,Win,incre)
t_  =np.linspace(1, Ts*np.size(y)*incre-Ts*incre, np.size(y))

#################################  minimos quadrados
PHI=np.zeros(( np.size(y),12))
PHI[0] = [0,          0,     0,       0,      0,      0,       0,      0,     0,      0,     0,     0 ]
PHI[1] =[-y[1],       0,     0,       0,     u2[1],    0,      0,      0,     u3[1],  0,     0,     0 ]
PHI[2] =[-y[2],     -y[1],   0,       0,     u2[2],  u2[1],    0,      0,     u3[2], u3[1],  0,     0 ]
PHI[3] =[-y[3],     -y[2],  -y[1],    0,     u2[3],  u2[2],   u2[1],   0,     u3[3], u3[2],  u3[1], 0]  


yest=y[20:130]
for k in range(np.size(yest)):
    PHI[k] = [-y[k-1], -y[k-2], -y[k-3], -y[k-4], u2[k-1], u2[k-2], u2[k-3], u2[k-4], u3[k-1], u3[k-2], u3[k-3], u3[k-4]];


theta = inv(PHI.T.dot(PHI)).dot(PHI.T.dot(y))

################################
 
ref_arquivo.close()

data=np.zeros(( np.size(y),12))
ys=np.zeros(( np.size(y),1))
phi=np.zeros(( 1,12))
for k in range( np.size(y)):
    #data[k]=[u2[k-1], u2[k-2], u2[k-3] ,u2[k-4], u3[k-1] ,u3[k-2], u3[k-3] ,u3[k-4]] # modelo autorregressivo  rede NAR
    phi[0]=[-ys[k-1], -ys[k-2], -ys[k-3], -ys[k-4], u2[k-1], u2[k-2], u2[k-3], u2[k-4], u3[k-1], u3[k-2], u3[k-3], u3[k-4]]
    data[k]=[-ys[k-1], -ys[k-2], -ys[k-3], -ys[k-4], u2[k-1], u2[k-2], u2[k-3], u2[k-4], u3[k-1], u3[k-2], u3[k-3], u3[k-4]]
    ys[k]=phi[0].dot(theta)
    
#################################  rede neural 
model = Sequential()
model.add(Dense(12, input_dim=12, activation='softplus')) # logsig
model.add(Dense(1, activation='linear'))

eta=0.1

optimizer = tf.keras.optimizers.Adam(eta) # algoritmo de treinamento ADAM
model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

model.fit(data,y, batch_size=20, epochs=50, shuffle=True)

yp=np.zeros(( np.size(y),1))
predict=yp;
for k in range( np.size(y)):
    
    data[k]=[-yp[k-1], -yp[k-2], -yp[k-3], -yp[k-4], u2[k-1], u2[k-2], u2[k-3], u2[k-4], u3[k-1], u3[k-2], u3[k-3], u3[k-4]]
    x=data[k];
    x.shape=(1,12)
    yp[k] = model.predict(x)
    
    predict[k]=yp[k]

###################################################### indices
R2_rede=functions.R_2(y,predict)
RME_rede=functions.MSE(y,predict)
R2_arx=functions.R_2(y,ys)
RME_arx=functions.MSE(y,ys)

plt.Figure()
plt.plot( t_ , y, 'r',t_, predict, 'b')
plt.title("R^2: "+str(R2_rede)+"%  EMQ: "+str(RME_rede) );
plt.show()

plt.Figure()
plt.plot( t_ , y, 'r',t_, ys, 'b')
plt.title("R^2: "+str(R2_arx)+"%  EMQ: "+str(RME_arx) );
plt.show()
