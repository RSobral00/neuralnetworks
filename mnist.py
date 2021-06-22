#Rúben Sobral- Universidade de Aveiro - Departamento de Física 
# Código retirado e adaptado de https://github.com/fchollet/deep-learning-with-python-notebooks
import keras
import keras.backend as K
from keras import regularizers
keras.__version__

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # import do data set mnist

# train labels e test labels (num de 0 a 9)


from keras import models
from keras import layers

model = models.Sequential()
funcactiv = 'relu'                                      # func ativação 
     
#model.add(layers.Dense(512, activation= funcactiv, input_shape=(28 * 28,),kernel_regularizer=keras.regularizers.l2(l=0.01)))   Se a pretenção for usar Regularizador, escolher o regularizador e a intensidade e substituir esta linha por a seguinte linha
model.add(layers.Dense(512, activation= funcactiv, input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax' ))                               

                                                            # Para mudar o momentum tem que ser espeficificado no otimizador ( ou seja mudar o tipo de otimizador)
opt = keras.optimizers.Adamax(learning_rate=0.01)         # Otimizador e Learning rate 

model.compile(optimizer= opt,                                  
                loss='categorical_crossentropy',                       
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255



from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

x_val = test_images[-10000:]
y_val = test_labels[-10000:]
x_train = train_images[:-10000]
y_train = train_labels[:-10000]

history = model.fit(x_train,
                    y_train,
                    epochs=200,
                    batch_size=128,
                    validation_data=(x_val, y_val))

print("-----Mnist-----")
print("Activation function:", funcactiv)
print("Regularization: l2")
print("Optimizer:" , model.optimizer)
print("Learning rate:" , K.eval(model.optimizer.lr))
#print("Momentum:")



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
plt.title('Adamax-LR=0.01-L2(0.01)-Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()




plt.figure(2)
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Adamax-LR=0.01-L2(0.01)-Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()