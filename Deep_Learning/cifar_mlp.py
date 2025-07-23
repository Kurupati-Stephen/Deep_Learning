# Author: Kurupati Stephen
# File: cifar_mlp.py
# Description: Simple MLP on CIFAR-10 dataset

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(units=10, activation='softmax'))

# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(x_train, y_train, epochs=10, batch_size=64,validation_split=0.2)
print(history.history.items())
print(history.history.keys())

loss,accuracy = model.evaluate(x_test, y_test)
print(f'accuracy: {accuracy}')

plt.plot(history.history['accuracy'], label='accuracy',color='blue')
plt.plot(history.history['val_accuracy'], label='val_accuracy', color='Red')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

