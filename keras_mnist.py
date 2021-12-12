from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 784))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 784))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu',use_bias=False, input_shape=(784,)))
network.add(layers.Dense(256, activation='relu',use_bias=False))
network.add(layers.Dense(10, activation='softmax',use_bias=False))

network.summary()

network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=14, batch_size=128)

network.evaluate(test_images,test_labels)