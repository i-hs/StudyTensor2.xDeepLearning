import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 데이터를 다운받습니다.
from tensorflow_core.python.keras.models import load_model


def train(tm):
    '''
    train(tm) : 학습에 필요한 필수 함수

    labels
        tm.features.get('y_value')
    data load
        tm.load_data_for_cnn()    # cnn 데이터
        tm.load_data_for_all()    # 기타 데이터
        ex) (train_id, train_x, train_y), (test_id, test_x, test_y) = tm.load_data_for_cnn()

    param
        tm.param_info[common_params]
        tm.param_info[algo_params]

    common_params
        nn_type          NN 유형
        init_method      초기화 방법
        opt_method       최적화 방법
        learning_rate    Learning Rate
        dropout_ratio    Dropout Ratio
        random_seed      랜덤 seed
        autosave_p       자동저장 주기
        epoch            학습수행횟수
        batch_size       배치 사이즈

    algo_params
        ui 에서 정의

    save
        tm.save_result_metrics(eval_results)

    eval_results
        step
        predict_y
        actual_y
        test_id
        confusion_matrix

    '''

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    # (50000, 32, 32, 3) (50000, 1)
    # (10000, 32, 32, 3) (10000, 1)

    class_names = ['airplane', 'automobile', 'bird', 'cat',
                  'deer', 'dog', 'frog', 'horse',
                  'ship', 'truck']

    # sample_size = 9
    # random_idx = np.random.randint(60000, size=sample_size)

    # 평균과 표준편차는 채널별로 구해줍니다.
    x_mean = np.mean(x_train, axis = (0, 1, 2))
    x_std = np.std(x_train, axis = (0, 1, 2))

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std


    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.3)

    print(x_train.shape, len(y_train))
    print(x_val.shape, len(y_val))

    # (35000, 32, 32, 3) 35000
    # (15000, 32, 32, 3) 15000

    model = Sequential()
    # (32, 32, 3)의 데이터를 입력으로 받습니다.
    model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (32, 32, 3)))
    model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), strides = 2, padding = 'same'))
    model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), strides = 2, padding = 'same'))
    model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), strides = 2, padding = 'same'))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax')) # 10개의 출력을 가지는 신경망

    model.compile(optimizer = Adam(1e-4),
                 loss = 'sparse_categorical_crossentropy',
                 metrics = ['acc'])
    # 4. 모델 학습하기
    history =model.fit(x_train, y_train,
                epochs=30,
                batch_size=32,
                validation_data=(x_val, y_val))

    # 5. 모델 학습과정 그리기
    model_graph(history)

    # 5. 모델 평가하기
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
    print('loss_and_metrics : ' + str(loss_and_metrics))

    # 6. 모델 저장하기
    model_path = tm.model_path
    model_name = tm.save_filename
    save_path = os.path.join(model_path, model_name)
    model.save(save_path)

def model_graph(history):

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

def load_model_analysis(im):

    global model

    # 2. 모델 불러오기

    model_path = im.model_path
    model_name = tm.save_filename
    mnist_model_path = os.path.join(model_path, model_name)
    model = load_model(mnist_model_path)

    # 테스트 데이터 가져오기
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    # (50000, 32, 32, 3) (50000, 1)
    # (10000, 32, 32, 3) (10000, 1)

    class_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse',
                   'ship', 'truck']

    # sample_size = 9
    # random_idx = np.random.randint(60000, size=sample_size)

    # 평균과 표준편차는 채널별로 구해줍니다.
    x_mean = np.mean(x_train, axis=(0, 1, 2))
    x_std = np.std(x_train, axis=(0, 1, 2))

    # x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    # 5. 모델 평가하기
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
    print('loss_and_metrics : ' + str(loss_and_metrics))

    # 6. 시각화하기

    get_layer_name = [layer.name for layer in model.layers]
    get_output = [layer.output for layer in model.layers]

    # 모델 전체에서 output을 가져올 수 있습니다.
    visual_model = tf.keras.models.Model(inputs=model.input, outputs=get_output)

    # 테스트셋의 두 번째 데이터는 '배'입니다.
    test_img = np.expand_dims(x_test[1], axis=0)
    feature_maps = visual_model.predict(test_img)

    for layer_name, feature_map in zip(get_layer_name, feature_maps):
        # Dense 층은 제외합니다.
        if (len(feature_map.shape) == 4):
            img_size = feature_map.shape[1]
            features = feature_map.shape[-1]

            display_grid = np.zeros((img_size, img_size * features))

            # 각 특징맵을 display_grid 배열에 이어붙입니다.
            for i in range(features):
                x = feature_map[0, :, :, i]
                x -= x.mean();
                x /= x.std()
                x *= 64;
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * img_size: (i + 1) * img_size] = x

            plt.figure(figsize=(features, 2 + 1. / features))
            plt.title(layer_name, fontsize=20)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()




class TrainManagerTmp():
    '''알고리즘과 플랫폼과의 연동을 도와주는 Helper class'''

    def __init__(self):
        # self.train_data_path = ""
        # self.param_info = {'learning_rate' : 0.0003000000142492354, 'autosave_p' : 5, 'batch_size' : 1}
        self.model_path = "./model"
        self.save_filename = "cifar_model.h5"


if __name__ == '__main__':
    tm = TrainManagerTmp()
    train(tm)
    # load_model_analysis(tm)
