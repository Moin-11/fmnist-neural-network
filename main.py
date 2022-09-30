import tensorflow as tf
from tensorflow import keras

fmnist = tf.keras.datasets.fashion_mnist

(tr_images, tr_labels), (test_images, test_labels) = fmnist.load_data()


# define call back class

class CallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4):
            print("\ntraining cancelled as 60% accuracy reached")
            self.model.stop_training = True


callbacks = CallBack()

# normalize data between 0 and 1
tr_images = tr_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, tf.nn.relu),
                          keras.layers.Dense(10, tf.nn.softmax)])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(tr_images, tr_labels, 5)

model.fit(tr_images, tr_labels, epochs=10, callbacks=[callbacks])
