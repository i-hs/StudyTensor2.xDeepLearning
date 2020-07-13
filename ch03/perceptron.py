import tensorflow as tf
tf.random.set_seed(777)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse

x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [1]])

# 모델 구성하기
model = Sequential()

# 단층 퍼셉트론 구성
model.add(Dense(1, input_shape=(2,), activation='linear'))

# 모델 준비하기
model.compile(optimizer=SGD(),
              loss=mse,
              metrics=['acc'])

# 학습시키기
model.fit(x, y, epochs=500)
print(model.get_weights())