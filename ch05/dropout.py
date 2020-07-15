import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.layers import Dropout
from tensorflow_core.python.keras.models import load_model
from tensorflow_core.python.keras.regularizers import l2


def train(tm):
    # 1. 데이터셋 생성하기
    # 훈련셋과 시험셋 불러오기
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 평균과 표준편차는 채널별로 구해줍니다.
    x_mean = np.mean(x_train, axis=(0, 1, 2))
    x_std = np.std(x_train, axis=(0, 1, 2))

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std


    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)
    print('data ready~')



    # 2. 모델 구성하기
    model = Sequential()
    # 입력 데이터는 (75, 75, 3)의 형태를 가집니다.
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(Dropout(0.2))  # 드롭아웃을 추가합니다.
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(Dropout(0.2))  # 드롭아웃을 추가합니다.
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(Dropout(0.2))  # 드롭아웃을 추가합니다.
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # 3. 모델 학습과정 설정하기
    model.compile(optimizer=Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    # 5. 모델 학습과정 그리기
    history = model.fit(x_train, y_train,
                        epochs=30,
                        batch_size=32,
                        validation_data=(x_val, y_val))

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



class TrainManagerTmp():
    '''알고리즘과 플랫폼과의 연동을 도와주는 Helper class'''

    def __init__(self):
        # self.train_data_path = ""
        # self.param_info = {'learning_rate' : 0.0003000000142492354, 'autosave_p' : 5, 'batch_size' : 1}
        self.model_path = "./model"
        self.save_filename = "dropout_model.h5"

if __name__ == '__main__':
    tm = TrainManagerTmp()
    train(tm)
    # load_model_eval(tm)