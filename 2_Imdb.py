#Rúben Sobral- Universidade de Aveiro - Departamento de Física 
# Código retirado e adaptado de https://github.com/fchollet/deep-learning-with-python-notebooks
import keras
import keras.backend as K
keras.__version__

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


from keras import models
from keras import layers

funcactiv = 'relu'                                              #func activação

model = models.Sequential()
   #regularizer 
model.add(layers.Dense(16, activation=funcactiv, input_shape=(10000,)))        
#model.add(layers.Dense(16, activation=funcactiv,kernel_regularizer=keras.regularizers.l1(l=0.01)))  Se a pretenção for usar Regularizador, escolher o regularizador e a intensidade e substituir esta linha por a seguinte linha
model.add(layers.Dense(16, activation=funcactiv))                        
model.add(layers.Dense(1, activation='sigmoid'))



opt = keras.optimizers.SGD(learning_rate=0.1, momentum = 0.9)             # OTIMIZADOR e learning rate
                                                                        # Momentum depende do otimizador 

model.compile(optimizer=opt,                          
              loss='binary_crossentropy',                   
              metrics=['accuracy'])


x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=200,
                    batch_size=512,
                    validation_data=(x_val, y_val))

print("-----Imdb-----")
print("Activation function:", funcactiv)
#print("Regularization: l1")
print("Optimizer:" , model.optimizer)
print("Learning rate:" , K.eval(model.optimizer.lr))
#print("Momentum: Não há")


import matplotlib.pyplot as plt

history_dict = history.history


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.figure(1)

plt.plot(epochs, loss, 'r', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('SGD-LR=0.1-Momentum=0.9-Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()




plt.figure(2)
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('SGD-LR=0.1-Momentum=0.9-Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
