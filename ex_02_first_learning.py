import collections  # used for counting items of a list
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from keras import layers
from keras.datasets import fashion_mnist
from tensorflow import keras
import os

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

"""
 the data, split between train and test sets
x_train: images for training
y_train: labels for training
x_test: images for testing the model
y_test: labels for testing the model
"""
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

"""
#download the images
"""
path = "/home/toalba/PycharmProjects/machine_learning_examples/picture/"

"""# Aufgabe 1.1.3"""
print("Pixel von Bild 10")
print(x_train[10])
print("storing images.....")
"""# Aufgabe 1.2.1"""
for i in range(0, 100):
    im = Image.fromarray(x_train[i])
    real = y_train[i]
    im.save(path + str(i) + ".jpeg")

for i in range(0,100):
    category = y_train[i]
    im = Image.fromarray(x_train[i])
    pathC = "/home/toalba/PycharmProjects/machine_learning_examples/picture/" + str(category)+"/"
    if not os.path.exists(pathC):
        os.mkdir(pathC)
    im.save(pathC + str(i) + ".jpeg")

"""
# Scale images to the [0, 1] range
# Cast to float values before to make sure result is float
"""
x_train = x_train.astype("float32") / 255
print(x_train.shape, "train samples")
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(x_train.shape, "x_train shape:")
print(x_train.shape[0], "number of train samples")
print(x_test.shape[0], "number of test samples")


"""# Aufgabe 1.1.2"""
nr_labels_y = collections.Counter(y_train)  # count the number of labels
print(nr_labels_y, "Number of labels")

# convert class vectors (the labels) to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_labels = y_test  # use this to leave the labels untouched
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
Build the model
"""

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

""" #Aufgabe 2 """
""" #Aufgabe 2.1 """
model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(56, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 3
"""# Aufgabe 2.0"""
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# Je mehr Epochen es gibt, desto genauer kann es werden,
# jedoch gibt es ein Limit, dass man die Daten verwerten kann.

# draw the learn function
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()
"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

"""
## Do some Predictions on the test dataset and compare the results
"""

pred = model.predict(x_test)

print(pred[1])  # Prediction for image 1
pred_1 = np.argmax(pred[1])
#print(pred_1)

for i in range(0, 100):
    pred_i = np.argmax(pred[i])  # get the position of the highest value within the list
 #   print(y_labels[i], pred_i)

"""
#How to load and save the model
"""

model.save('/home/albert/model.mdl')
model.save_weights("/home/albert/model.h5")

weights = model.get_weights()
j = json.dumps(pd.Series(weights).to_json(orient='values'), indent=3)
#print(j)

model = keras.models.load_model('/home/toalba/PycharmProjects/machine_learning_examples/modle/model.mdl')
model.load_weights("/home/toalba/PycharmProjects/machine_learning_examples/modle/model.h5")

model_json = model.to_json()
#print(model_json)
