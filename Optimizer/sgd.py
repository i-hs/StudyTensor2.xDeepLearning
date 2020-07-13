import numpy as np
import matplotlib.pyplot as plt

class Sgd:
    """ SGD: Stochastic Gradient Descent
    W = W - lr * dL/dW
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        for key in params:
            # W = W - lr * dl/dW
            params[key] -= self.learning_rate * gradients[key]



def fn(x, y):
    """f(x, y) = (1/20) * x**2 + y**2"""
    return x**2 / 20 + y**2

def fn_derivative(x, y): # 함수 fn의 미분값
    return x/10, 2*y


if __name__ == '__main__':
    # Sgd 클래스의 객체(인스턴스)를 생성
    sgd = Sgd(0.95)

    # ex01 모듈에서 작성한 fn(x, y) 함수의 최솟값을 임의의 점에서 시작해서 찾아감.
    init_position = (-7, 2)

    # 신경망에서 찾고자 하는 파라미터의 초깃값
    params = dict()
    params['x'], params['y'] = init_position[0], init_position[1]

    # 각 파라미터에 대한 변화율(gradient)
    gradients = dict()
    gradients['x'], gradients['y'] = 0, 0

    # 각 파라미터들(x, y)을 갱신할 때마다 갱신된 값을 저장할 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])  # gradients 갱신
        sgd.update(params, gradients)


    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')

    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    plt.contour(X, Y, Z, 30)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    # 등고선 그래프에 파라미터(x, y)들이 갱신되는 과정을 추가.
    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()