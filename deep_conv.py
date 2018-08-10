import tensorflow as tf
import keras
import pandas as pd

train_file="C:/Users/sindh/Documents/Python/train.csv"
raw_data = pd.read_csv(train_file)

img_rows, img_cols = 28,28
num_classes=10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

x, y= data_prep(raw_data)

model = keras.Sequential()
model.add(tf.keras.layers.Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(tf.keras.layers.Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])	
model.fit(x, y, batch_size=128, epochs=2, validation_split=0.2)