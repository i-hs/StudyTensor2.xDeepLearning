# -*- coding: utf-8 -*-
"""mnist_cnn_fits

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jgU4p7BOFOPxeZb7_j-RrcUBT7lBhuIz

1.  GPU
2. 버전, 사용여부
3. 데이터 전처리
4. 신경망 생성 </br>
Conv2D -> Conv2D -> MaxPool2D -> Dropout -> Flatten -> Dense -> Dropout -> Dense
page 262
Conv2D : filters=32, kernel_size=(3, 3), act=relu </br>
Conv2D : filters=64, kernel_size=(3, 3), act=relu </br>
Dropout: dropout 비율 = 0.25(25%) </br>
Dense : 퍼셉트론 개수=128, act=relu </br>
Dropout: dropout 비율 = 0.5 </br>
Dense : 퍼셉트론 개수=10, act=softmax </br>
5. 모델 학습(fit)
6. epoch-loss, epoch-accuracy 그래프 그리기
"""

# Commented out IPython magic to ensure Python compatibility.
# Tensorflow 버전
# %tensorflow_version 2.x

# Tensorflow 버전과 GPU 사용여부 확인
import tensorflow as tf

print(tf.__version__)
print(tf.test.gpu_device_name())

# 클래스 및 모듈 import
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Mnist 전처리
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(f'X_train:{X_train.shape}, Y_train:{Y_train.shape}')
print(f'X_test:{X_test.shape}, Y_test:{Y_test.shape}')

# X Shape 전환
X_train = X_train.reshape(*X_train.shape, 1)
X_test = X_test.reshape(*X_test.shape, 1)
# Y shape 전환
Y_train = to_categorical(Y_train, 10, 'float16')
Y_test = to_categorical(Y_test, 10, 'float16')

# X_train, X_test 정규화
X_train = X_train.astype('float16')/255
X_test = X_test.astype('float16')/255

# shape 확인
print(f'X_train:{X_train.shape}, Y_train:{Y_train.shape}')
print(f'X_test:{X_test.shape}, Y_test:{Y_test.shape}')

# 신경망 모델 생성
model = Sequential()

# 신경망 모델에 은닉층, 출력층 계층들을 추가
"""Conv2D -> Conv2D -> MaxPool2D -> Dropout -> Flatten -> Dense -> Dropout -> Dense page 262 
Conv2D : filters=32, kernel_size=(3, 3), act=relu
Conv2D : filters=64, kernel_size=(3, 3), act=relu
Dropout: dropout 비율 = 0.25(25%)
Dense : 퍼셉트론 개수=128, act=relu
Dropout: dropout 비율 = 0.5
Dense : 퍼셉트론 개수=10, act=softmax"""

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 신경망 모델의 성능 향상이 없는 경우 중단시키기 위해서
early_stop = EarlyStopping(monitor='val_loss',
                           verbose=1,
                           patience=10)

# 신경망 학습
history = model.fit(X_train, Y_train,
                    batch_size=200,
                    epochs=50,
                    verbose=1,
                    callbacks=[early_stop],
                    validation_data=(X_test,Y_test))

train_loss=history.history['loss']
test_loss=history.history['val_loss']

x = range(len(train_loss))
plt.plot(x, train_loss, marker='.', color='red', label='Train loss')
plt.plot(x, test_loss, marker='.', color='blue', label='Test loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

train_acc = history.history['acc']  # list type
test_acc = history.history['val_acc']  # list type

x = range(len(train_loss))
plt.plot(x, train_acc, marker='.', color='red', label='Train accuracy')
plt.plot(x, test_acc, marker='.', color='blue', label='Test accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()