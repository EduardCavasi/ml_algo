import struct
import numpy as np
import os
import matplotlib.pyplot as plt
from keras import layers
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.metrics.metrics_utils import confusion_matrix
from tensorflow import keras


def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def show_data(x, y):
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i], cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_wrong_prediction(x, y, y_pred):
    plt.figure(figsize=(10, 4))
    k = 1
    for i in range(10000):
        if y_pred[i] != y[i]:
            plt.subplot(10, 10, k)
            k = k + 1
            plt.imshow(x[i], cmap='gray')
    plt.tight_layout()
    plt.show()

def tensorflow_preprocessing(x):
    datagen = ImageDataGenerator(
        rotation_range=10,  # Randomly rotate images by up to 10 degrees
        width_shift_range=0.1,  # Randomly shift images horizontally by up to 10%
        height_shift_range=0.1,  # Randomly shift images vertically by up to 10%
        zoom_range=0.1,  # Randomly zoom images by up to 10%
        shear_range=0.1,  # Apply shearing transformations
        fill_mode='nearest'  # Fill in newly created pixels
    )
    datagen.fit(x)

def tensorflow_model(x_train, y_train, x_test, y_test):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001,
                                                   restore_best_weights=True)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=128,
              callbacks=[early_stopping])

    y_pred = model.predict(x_test)
    labels_pred = np.argmax(y_pred, axis=1)

    show_data(x_test, labels_pred)
    show_wrong_prediction(x_test, y_test, labels_pred)
    print(confusion_matrix(y_test, np.argmax(model.predict(x_test), axis=1), num_classes=10))

data_path = "./data/MNIST/raw/"
x_train = load_images(os.path.join(data_path, "train-images.idx3-ubyte"))
y_train = load_labels(os.path.join(data_path, "train-labels.idx1-ubyte"))
x_test = load_images(os.path.join(data_path, "t10k-images.idx3-ubyte"))
y_test = load_labels(os.path.join(data_path, "t10k-labels.idx1-ubyte"))

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
tensorflow_preprocessing(x_train)
#show_data()

tensorflow_model(x_train, y_train, x_test, y_test)