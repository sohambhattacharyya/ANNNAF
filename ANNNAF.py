# -*- coding: utf-8 -*-
"""Artifiial Neural Network for Nonclassical Adaptive Filtering"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units=128, activation='relu', input_dim=512))
model.add(Dense(units=128))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# Load training and eval data
t = np.arange(512)
x_t=np.float32(5*t)
#ip=np.stack((t,x_t),axis=0)
ip=np.expand_dims(x_t, axis=0)
d_t=2*t**3+3*t**2
x_t_eval=x_t[:256]
y_t_eval=d_t[:256]
op=np.stack((t[:256],x_t_eval),axis=0)

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(ip, d_t, epochs=10, batch_size=128)

y_t = model.predict(x_t_eval, batch_size=128)
