# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 13:26:07 2018

@author:
    Bruno Hjort
    Emanoel Eberle Kruger
    Orion Silva
"""

from keras import models, layers

model = models.Sequential()
model.add(layers.Conv2D(16, (3,3),activation='relu', input_shape=(100,100,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(16, (3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Conv2D(128, (3,3),activation='relu'))
#model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(3, activation = 'relu'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


from keras.preprocessing.image import ImageDataGenerator
import os
base_dir = "G:\\Pós\\Visão Computacional\\Aula 4\\Treinamento3"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir,"Treina"),
        target_size=(100,100),
        batch_size=2,
        class_mode="categorical",
        classes=['Brabo','Feliz','Neutro']
        )

validation_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir,"Valida"),
        target_size=(100,100),
        batch_size=2,
        class_mode="categorical",
        classes=['Brabo','Feliz','Neutro']
        )

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 5,
        validation_data = validation_generator,
        validation_steps = 50
        )

model.save(base_dir+'\\Humor_Emanoel.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
