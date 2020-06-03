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
    print('Attempting to connect to PostgreSQL...')
    conn = psycopg2.connect(host='localhost', database='origami', user='postgres', password='postgres')
    cur = conn.cursor()
    # SELECT image classifications and file paths from PostgreSQL
    sql_select = "SELECT image_class, image_file FROM origami_images;"
    cur.execute(sql_select)
    print('Selecting rows from origami_images table...')
    origami_images = cur.fetchall()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        cur.close()
        conn.close()
        print('PostgreSQL connection closed.')



# IMAGE PROCESSING

# Function to get a category from a file path
def get_cat(file_path):
    cat = '' #TODO: return once file repository is set up and structure is clear
    return cat

# Function to decode an image
def decode_img(img, img_size):
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [img_size, img_size])

# Function to process a file path and return the image
def process_path(file_path, img_size):
    category = get_cat(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_size)
    return img, category



# Normalize images and categories
# TODO: convert to grayscale, standardize dimensions (256x256?), normalize pixel values, convert labels to num categorical
image_count = 0 #TODO: replace w function
batch_size = 32
std_img_size = 256
steps_per_epoch = np.ceil(image_count / batch_size)
num_cat = 5 #TODO: replace w dynamic function w list or dict

# Segment into train and test sets
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
# TODO: export model to an HDF5 file