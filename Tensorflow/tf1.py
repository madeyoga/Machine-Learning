import tensorflow as tf

mnist = tf.keras.datasets.mnist #28x28 image of handwritten digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
##plt.imshow(x_train[0])
#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()
#print(x_train[0])


## scale/normalize data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()

import os.path

if os.path.exists('epic_num_reader.model'):
    model = tf.keras.models.load_model('epic_num_reader.model')
else:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten()) # input layer
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) ## hidden layer
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) ## hidden layer
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) ## output layer
    # 128, unit value. 10, output class
  
    model.compile(
        optimizer='adam', # 'stochastic' gradient descent
        loss='sparse_categorical_crossentropy', # degree of error, 
        metrics=['accuracy']
        )

    model.fit(
        x_train,
        y_train,
        epochs=3
        )
        
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
# model.save('epic_num_reader.model')

import numpy as np
predictions = model.predict([x_test])
print("Prediction: " + str(np.argmax(predictions[0])))
plt.imshow(x_test[0])
plt.show()
