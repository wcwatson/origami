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

# Get list of class names
class_names = []
for img in origami_images:
    if img[0] not in class_names:
        class_names.append(img[0])
class_names = np.array(class_names)

# Generate tf Dataset from list of paths
path_ds = tf.data.Dataset.from_tensor_slices(origami_images)

# Side length of normalized image
IMG_SIZE = 256
# AUTOTUNE parameter
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Other parameters
img_count = len(origami_images)
BATCH_SIZE = 32
STEPS_PER_EPOCH = np.ceil(img_count/BATCH_SIZE)

# AUXILIARY FUNCTIONS TO PROCESS IMAGES
# Function to return class label
def get_label(category):
    return category == class_names

# Function to decode an image, render in grayscale and square dimensions
def decode_img(img):
    img = tf.io.decode_image(img, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.rgb_to_grayscale(img)
    return tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

# Function to process a file path and return the image
def process_path(img_tuple):
    # Get label from auxiliary function
    label = get_label(img_tuple[0])
    # Decode image using auxiliary function
    file_path = img_tuple[1]
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

# Generate datset of images, batch it
image_ds = path_ds.map(process_path, num_parallel_calls=AUTOTUNE)
image_ds = image_ds.batch(BATCH_SIZE)
image_ds = image_ds.prefetch(1)


# DEFINE AND COMPILE MODEL

# Model definition
nn = keras.Sequential()
# Layer construction w/ L2 regularizers on convolutional layers
nn.add(keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
nn.add(keras.layers.MaxPooling2D((2, 2)))
nn.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
nn.add(keras.layers.MaxPooling2D((2, 2)))
nn.add(keras.layers.Flatten())
nn.add(keras.layers.Dense(16, activation='relu'))
nn.add(keras.layers.Dense(len(class_names), activation='softmax'))

# Model compilation
nn.compile(optimizer='adam',
           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           metrics=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    'categorical_accuracy'])

# Model training
nn.fit(image_ds, epochs=10)

# TODO: train the model on the read-in data, use an automatic stop to prevent overfitting

# Export model for future use
# TODO: export model to pickle