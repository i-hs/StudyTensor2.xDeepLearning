from tensorflow_core.python.keras.layers import Dense
from tensorflow_core.python.keras.models import Sequential

model = Sequential()
model.add(Dense(32, input_shape=(2, ), activation='relu'))
model.add(Dense(1, activation='sigmoid'))