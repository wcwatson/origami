# Convoluational neural network for classifying images of origami

# Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import psycopg2
from PIL import Image
import os
import h5py

# Read in data from origami database in PostgreSQL
try:
    print('Attempting to connect to PostgreSQL.')
    conn = psycopg2.connect(host='localhost', database='origami', user='postgres', password='postgres')
    cur = conn.cursor()
    # SELECT image classifications and file paths from PostgreSQL
    sql_select = "SELECT image_class, image_path FROM origami_images;"
    cur.execute(sql_select)
    print('Selecting rows from origami_images table.')
    origami_images = cur.fetchall()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        cur.close()
        conn.close()
        print('PostgreSQL connection closed.')
#print(origami_images)

#TODO: Piece back together from jupyter notebook
'''
# Function to decode an image
def decode_img(img, img_size):
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [img_size, img_size])

# Function to process a file path and return the image
def process_path(img_tuple, img_size):
    category = img_tuple[0]
    file_path = img_tuple[1]
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_size)
    return img, category


def compile_and_fit(model, max_epochs=10000):
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='binary_crossentropy'),
                           'accuracy'])
    model.summary()
    history = model.fit(
        train_set,
        #steps_per_epoch = STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=val_set,
        verbose=1)
    return history


# Normalize images and categories
# TODO: convert to grayscale, standardize dimensions (256x256?), normalize pixel values, convert labels to num categorical
image_count = 0 #TODO: replace w function
batch_size = 32
std_img_size = 256
steps_per_epoch = np.ceil(image_count / batch_size)
num_cat = 5 #TODO: replace w dynamic function w list or dict

# Segment into train and test sets
train_set = []
val_set = []
test_set = []
# TODO: fill in, easy enough...


# Model initialization
nn = keras.Sequential()
# Layer construction w/ L2 regularizers on convolutional layers
nn.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),
                           input_shape=(std_img_size, std_img_size, 1)))
nn.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
nn.add(keras.layers.Flatten())
nn.add(keras.layers.Dense(num_cat, activation='softmax'))

# Model training
# TODO: train the model on the read-in data, use an automatic stop to prevent overfitting

# Export model for future use
# TODO: export model to pickle
'''