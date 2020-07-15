import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from tensorflow_core.python.keras.models import load_model


def train(tm):
    # 1. 데이터셋 생성하기

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 평균과 표준편차는 채널별로 구해줍니다.
    x_mean = np.mean(x_train, axis=(0, 1, 2))
    x_std = np.std(x_train, axis=(0, 1, 2))

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std


    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)
    print('data ready~')

    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                       zoom_range=0.2,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       rotation_range=30,
                                       fill_mode='nearest')

    val_datagen = ImageDataGenerator()

    batch_size = 32

    train_generator = train_datagen.flow(x_train, y_train,
                                         batch_size=batch_size)

    val_generator = val_datagen.flow(x_val, y_val,
                                     batch_size=batch_size)

    # 2. 모델 구성하기
    # imagenet을 학습한 모델을 불러옵니다.
    vgg16 = VGG16(weights='imagenet', input_shape=(32, 32, 3), include_top=False)
    vgg16.summary()

    # 끝의 4개의 층만 동결을 해제합니다.
    for layer in vgg16.layers[:-4]:
        layer.trainable = False

    model = Sequential()
    model.add(vgg16)
    # 분류기를 직접 정의합니다.
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))

    # model.summary() # 모델의 구조를 확인하세요!





    # 3. 모델 학습과정 설정하기

    model.compile(optimizer=Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    # 5. 모델 학습과정 그리기

    history = model.fit(train_generator,
                        epochs=100,
                        steps_per_epoch=get_step(len(x_train), batch_size),
                        validation_data=val_generator,
                        validation_steps=get_step(len(x_val), batch_size))
    ###

   # 5. 모델 평가하기

    his_dict = history.history
    loss = his_dict['loss']
    val_loss = his_dict['val_loss']

    epochs = range(1, len(loss) + 1)
    fig = plt.figure(figsize=(10, 5))

    # 훈련 및 검증 손실 그리기
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, loss, color='blue', label='train_loss')
    ax1.plot(epochs, val_loss, color='orange', label='val_loss')
    ax1.set_title('train and val loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend()

    acc = his_dict['acc']
    val_acc = his_dict['val_acc']

    # 훈련 및 검증 정확도 그리기
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, acc, color='blue', label='train_acc')
    ax2.plot(epochs, val_acc, color='orange', label='val_acc')
    ax2.set_title('train and val acc')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('acc')
    ax2.legend()

    plt.show()

    # 6. 모델 저장하기
    model_path = tm.model_path
    model_name = tm.save_filename
    save_path = os.path.join(model_path, model_name)
    model.save(save_path)



def get_step(train_len, batch_size):
    if(train_len % batch_size > 0):
        return train_len // batch_size + 1
    else:
        return train_len // batch_size

class TrainManagerTmp():
    '''알고리즘과 플랫폼과의 연동을 도와주는 Helper class'''

    def __init__(self):
        # self.train_data_path = ""
        # self.param_info = {'learning_rate' : 0.0003000000142492354, 'autosave_p' : 5, 'batch_size' : 1}
        self.model_path = "./model"
        self.save_filename = "imagenet_model.h5"


if __name__ == '__main__':
    tm = TrainManagerTmp()
    train(tm)
    # load_model_analysis(tm)