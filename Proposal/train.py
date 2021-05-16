import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
import os


def autoencoder_model(input_shape=(150, 150, 3)):
    kernel_size = (2, 2)
    pool_size = (2, 2)
    input_img = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size, activation='relu',
                      padding='same')(input_img)
    x = layers.MaxPooling2D(pool_size, padding='same')(x)
    x = layers.Conv2D(32, kernel_size, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size, padding='same')(x)
    x = layers.Conv2D(32, kernel_size, activation='relu', padding='same')(x)

    encoded = layers.MaxPooling2D(pool_size, padding='same', name='encoder')(x)

    x = layers.Conv2D(32, kernel_size, activation='relu',
                      padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, kernel_size, activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, kernel_size, activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(input_shape[2], kernel_size,
                            activation='sigmoid', padding='same')(x)

    autoencoder = k.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc'])

    return autoencoder


def model(encoder):
    encoder.trainable = False
    model = k.Sequential([
        encoder,
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(8, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    return model


if __name__ == '__main__':

    ds = tfds.load('colorectal_histology', split='train',
                   shuffle_files=True, as_supervised=True)
    input_shape = (150, 150, 3)
    images, labels = [], []

    for image, label in tfds.as_numpy(ds):
        images.append(image)
        labels.append(label)

    labels = np.array(labels)
    images = np.array(images, dtype='float')

    labels_cat = tf.keras.utils.to_categorical(labels, len(np.unique(labels)))
    labels_cat = np.array(labels_cat, dtype='uint8')

    datapoints = len(labels)
    training_split = 0.7

    x_train = images[:int(training_split * datapoints)]
    y_train = labels_cat[:int(training_split * datapoints)]

    x_test = images[int(training_split * datapoints):]
    y_test = labels_cat[int(training_split * datapoints):]

    print("[TRAIN]: (X,Y)", x_train.shape, y_train.shape)
    print("[TEST]: ", x_test.shape, y_test.shape)

    model_name = './models/autoencoder.h5'

    if not os.path.isfile(model_name):
        tensorboard_callback = k.callbacks.TensorBoard(
            log_dir="./logs/autoencoder")
        checkpoints = k.callbacks.ModelCheckpoint(model_name, monitor='loss', verbose=1,
                                                  save_best_only=True, mode='auto', period=1)

        autoencoder = autoencoder_model()

        autoencoder.summary()

        autoencoder.fit(x_train, x_train,
                        epochs=10,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(x_test, x_test), callbacks=[tensorboard_callback, checkpoints])

    tensorboard_callback = k.callbacks.TensorBoard(log_dir="./logs/classifier")
    checkpoints = k.callbacks.ModelCheckpoint('./models/classifier.h5', monitor='loss', verbose=1,
                                              save_best_only=True, mode='auto', period=1)

    # Obtain features from middle layer
    autoencoder = k.models.load_model(model_name)

    encoder = k.Model(inputs=autoencoder.input,
                      outputs=autoencoder.get_layer('encoder').output)

    classifier = model(encoder)

    classifier.fit(x_train, y_train,
                   epochs=10,
                   batch_size=32,
                   shuffle=True,
                   validation_data=(x_test, y_test), callbacks=[tensorboard_callback, checkpoints])
