import tensorflow as tf
from tensorflow_core.python.keras.losses import mse

tf.random.set_seed(777)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
label = np.array([[0], [1], [1], [0]])

# 모델 구성하기
model = Sequential()
model.add(Dense(32, input_shape=(2,), activation='relu'))

# 다층 퍼셉트론을 구성합니다.
model.add(Dense(1, activation='sigmoid'))

# 모델 준비하기
model.compile(optimizer=RMSprop(),
              loss=mse,
              metrics=['acc'])

# 학습시키기
model.fit(data, label, epochs=100)
