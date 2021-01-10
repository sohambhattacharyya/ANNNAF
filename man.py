# -*- coding: utf-8 -*-
"""N-hidden-layer Artifiial Neural Network for Nonclassical Adaptive Filtering"""

import numpy as np
import matplotlib.pyplot as plt

# Load training and eval data
#length=4096
#t = np.arange(length)

t=np.linspace(0,1,10)
d_t=0.2+0.4*t**2+0.3*t*np.sin(15*t)+0.05*np.cos(50*t)
length=t.size
x_t=np.float32(5*t)
ip=np.stack((t,x_t),axis=0)
#d_t=2*t**3+3*t**2
x_t_eval=t
y_t_eval=d_t
op=np.stack((t[:512],x_t_eval),axis=0)

# model specifications
Ni=1; Nh=4; No=1;
#parameter and array initialization
epochs=100
wh=np.random.randn(Nh,Ni); dwh=np.zeros(wh.shape) 
wo=np.random.randn(No,Nh); dwo=np.zeros(wo.shape) 
op=np.array([])
error=np.array([])

for epoch in range(epochs):
    for batch in range(np.int_(length/Ni)):        
        X=t[batch*Ni:(batch+1)*Ni]
        Y=d_t[batch*Ni:(batch+1)*Ni]
        h=1/(1+np.exp(-wh@X)) #hidden activation for all pattern
        yout=-wo@h
        y=1/(1+np.exp(yout)) #output for all pattern
        op=np.append(op,y)

        do=y*(1-y)*(Y-y)  # delta output
        dh=h*(1-h)*(wo.transpose()@do)  # delta backpropagated  
            
        # update weights with momentum
        dwh=0.9*np.outer(dh,X) 
        wh=wh+0.1*dwh
        dwo=0.9*dwo+np.outer(do,h)
        wo=wo+0.1*dwo
    
    error=np.append(error,np.sum(abs(Y-y)))

plt.xlabel("Iteration")
plt.ylabel("Absolute difference")
plt.plot(error)
plt.show()