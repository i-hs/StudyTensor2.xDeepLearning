import numpy as np
import matplotlib.pyplot as plt

lr_list = [0.001, 0.1, 0.3, 0.9]  # 여러가지 학습률을 사용하여 값의 변화를 관찰한다.'


def get_derivative(lr):
    w_old = 2
    derivative = [w_old]
    y = [w_old ** 2]  # 손실함수 x^2
    # 먼저 해당위치에서 미분값을 정의
    for i in range(1, 10):
        dev_value = w_old * 2

        # 가중치 업데이트
        w_new = w_old - lr * dev_value
        w_old = w_new

        derivative.append(w_old)  # 업데이트 된 가중치 내놔
        y.append(w_old ** 2)  # 업데이트된 가중치의 손실 값을 지정한다.

    return derivative, y


x = np.linspace(-2, 2, 50)  # -2 ~ 2 범위를 50구간으로 나눈 배열을 반환한다.
x_square = [i ** 2 for i in x]
fig = plt.figure(figsize=(12, 7))

for i, lr in enumerate(lr_list):
    derivative, y = get_derivative(lr)
    print("derivative: ", derivative)
    ax = fig.add_subplot(2, 2, i + 1)  # nrows, ncols, index, **kwargs
    ax.scatter(derivative, y, color='red')
    ax.plot(x, x_square)
    ax.title.set_text('lr = ' + str(lr))
plt.show()

