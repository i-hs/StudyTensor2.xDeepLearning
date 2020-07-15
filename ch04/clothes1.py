import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow_core.python.keras.layers import Flatten, Dense
from tensorflow_core.python.keras.models import Sequential

DATA_PATH = '../../clothes_dataset'
train_df = pd.read_csv(DATA_PATH + '/clothes_classification_train.csv')
val_df = pd.read_csv(DATA_PATH + '/clothes_classification_val.csv')
test_df = pd.read_csv(DATA_PATH + '/clothes_classification_test.csv')

print(train_df.head())

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 제너레이터를 정의합니다.
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)


def get_steps(num_samples, batch_size):
    if (num_samples % batch_size) > 0:
        return (num_samples // batch_size) + 1
    else:
        return num_samples // batch_size



model = Sequential()
# 입력 데이터 형태를 반드시 명시할 것
model.add(Flatten(input_shape=(112, 112, 3)))  # (112, 112, 3) -> (112 * 112 * 3)
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(11, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# 데이터 제너레이터 정의하기
batch_size = 32
class_col = ['black', 'blue', 'brown', 'green', 'red', 'white',
             'dress', 'shirt', 'pants', 'shorts', 'shoes']



# Make Generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='../../',
    x_col = 'image',
    y_col = class_col,
    target_size = (112, 112),
    color_mode='rgb',
    class_mode='other',
    batch_size=batch_size,
    seed=42
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='../../',
    x_col = 'image',
    y_col = class_col,
    target_size = (112, 112),
    color_mode='rgb',
    class_mode='other',
    batch_size=batch_size,
    shuffle=True
)

# 모델 학습 시키기
model.fit(train_generator,
         steps_per_epoch=get_steps(len(train_df), batch_size),
         validation_data = val_generator,
         validation_steps=get_steps(len(val_df), batch_size),
         epochs = 10)


# 5. 모델 평가하기
test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='../../',
    x_col='image',
    y_col=class_col,
    target_size=(112, 112),
    color_mode='rgb',
    class_mode='other',
    batch_size=batch_size,
    shuffle=True
)

loss_and_metrics = model.evaluate(test_generator,
                                  steps=get_steps(len(test_df), batch_size))
print('loss_and_metrics : ' + str(loss_and_metrics))


# 6. 모델 저장하기

save_path = os.path.join('model', 'img_generator_model.h5')
model.save(save_path)

# 책에는 명시되어 있지 않습니다.
preds = model.predict(test_generator,
                     steps = get_steps(len(test_df), batch_size),
                     verbose = 1)
# 8개만 예측해보도록 하겠습니다.
do_preds = preds[:8]

for i, pred in enumerate(do_preds):
    plt.subplot(2, 4, i + 1)
    prob = zip(class_col, list(pred))
    # item --> prob
    # contributor: '뱅커'님
    prob = sorted(list(prob), key=lambda z: z[1], reverse=True)[:2]

    image = cv2.imread(test_df['image'][i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title(f'{prob[0][0]}: {round(prob[0][1] * 100, 2)}% \n {prob[1][0]}: {round(prob[1][1] * 100, 2)}%')

plt.tight_layout()