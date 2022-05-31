# 딥러닝 모델 구현 코드의 기본 형태
# 사용 언어: 파이썬
# 딥러닝 라이브러리: Tensorflow에 포함된 Keras

import numpy as np  # 다양한 수치 계산에 관련된 기능 제공

from tensorflow import keras  # Tensorflow 라이브러리에서 keras 모듈을 import
from tensorflow.keras import optimizers  # Tensorflow 라이브러리의 keras 모듈에서 오류 보정을 통해 가중치를 갱신하기 위한 함수로 사용되는 SGD 등이 구현된 optimizers 모듈을 import
from tensorflow.keras.layers import Dense, Input  # Tensorflow 라이브러리의 keras 모듈에 포함된 layers 모듈에서 Dense와 Input 이라는 레이어의 구현 모듈을 import
# Dense: 완전연결층을 구현한 레이어 모듈
# Input: 입력을 위한 레이어 모듈
from matplotlib import pyplot as plt

model = keras.Sequential()  # keras가 제공하는 Sequential 이라는 클래스로 model 이라는 변수를 만든다. 이 model 이 구현할 딥러닝 모델의 형태가 된다.
print("model:", model)
model.add(Input(1))
model.add(Dense(10), activation='tanh')
model.add(Dense(10), activation='tanh')
model.add(Dense(1))

model.compile(optimizers="SGD", loss="mse")  # 구성한 모델을 사용할 수 있도록 컴파일을 시킨다. 어떤 최적화함수를 사용하고 어떤 손실함수를 사용할 것인지 지정하여 model을 바로 동작시킬 수 있도록 만드는 것이다.

x = np.arange(-1, 1, 0.01)  # -1부터 1 사이의 값을 0.01 간격으로 배열(array)을 생성하여 사용한다.
y = x**2  # 2차 방정식

model.fit(x, y, epoch=1000, verbose=0, batch_size=20)  # 모델의 학습은 fit 함수 사용

result_y = model.predict(x)  # model에 x라는 데이터를 넣고 그 결과를 예측해보라는 명령

# plt.scatter(x, y)
# plt.scatter(x, result_y, color='r')
# plt.show()
