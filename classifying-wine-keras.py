import numpy as np 
from data_proccesing import DataProcessing
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils import to_categorical
from keras.optimizers import Adam 
from keras. initializers import TruncatedNormal 
import matplotlib.pyplot as plt


#data processing 

dp = DataProcessing()
dp.processing('dataset/wine.data.txt', 14)

X_train, X_test = np.load('dataset/X_train.npy'), np.load('dataset/X_test.npy')
Y_train, Y_test = np.load('dataset/Y_train.npy'), np.load('dataset/Y_test.npy')


#binary array
Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)

#for model
input_dim = len(X_train[0, :])
class_num = len(Y_train[0, :]) 

model = Sequential()
init = TruncatedNormal(stddev=0.01, seed=10)

#config model 
model.add(Dense(units=50, activation='relu', input_dim=input_dim, kernel_initializer=init))
model.add(Dense(units=class_num, activation='softmax', kernel_initializer=init))

#optimizer
adam = Adam(lr=0.007)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#fit
history = model.fit(X_train, Y_train, epochs=800, validation_data=(X_test, Y_test), shuffle=False, verbose=0)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()