#Simple regression code

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense 

#data
X = np.linspace(0,10,320)
y = 3*X+7+3*np.random.randn(320)

#define architecture
model = Sequential()
model.add(Dense(units=1,input_dim=1,activation='linear'))

#compile
model.compile(optimizer='sgd',loss='mse', metrics=['mae'])

#train
model.fit(X,y,epochs=10,batch_size=16)


model.evaluate(X,y)
result = model.predict(X)



plt.scatter(X,y,label='Original data')
plt.plot(X,result,color='red',label = 'Predicted data')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
# plt.savefig("4.png")
plt.show()




