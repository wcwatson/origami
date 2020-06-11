# CNN for classifying images of origami
#
# NB: This is the code that could be run on my local machine to generate a model, but in actuality was not, because
# my machine is a 2014 MacBook Air and does NOT have the memory/processing power for that. Google Colab does, and so
# that's what I used to actually train the thing.
# TODO: in a future week, actually make this aligne with the model that I end up using

# Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import h5py


# READ IN DATA FROM ORIGAMI DATABASE IN POSTGRESQL
try:
    print('Attempting to connect to PostgreSQL.')
    user = 'wwatson'
    host = 'localhost'
    dbname = 'origami'
    # Connect to database
    db = create_engine('postgres://%s%s/%s'%(user, host, dbname))
    con = None
    con = psycopg2.connect(database=dbname, user=user)
    # Execute query
    print('Connection established. Executing query.')
    sql_query = "SELECT image_path AS filename, image_class AS class FROM origami_images;"
    query_results = pd.read_sql_query(sql_query, con)
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if con is not None:
        con.close()
        print('PostgreSQL connection closed.')


# LOAD IMAGES INTO TF GENERATORS

# Initialize ImageDataGenerator object, normalizing pixel values as floats in [0, 1]
train_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                             width_shift_range=0.2,
                                                             height_shift_range=0.2,
                                                             rescale=1./255,
                                                             shear_range=0.2,
                                                             zoom_range=0.2,
                                                             horizontal_flip=True,
                                                             fill_mode='nearest',
                                                             validation_split=0.2)
# For later TODO: actually use a test set...
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Set batch size, image resolution, and steps per epoch
IMG_SIZE = 64
BATCH_SIZE = 32
image_count = query_results.shape[0]
class_names = query_results['class'].unique()
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

# Generate training data from images
train_generator = train_datagen.flow_from_dataframe(query_results,
                                                    x_col='filename',
                                                    y_col='class',
                                                    target_size=(IMG_SIZE, IMG_SIZE, 3),
                                                    color_mode='rgb',
                                                    batch_size=BATCH_SIZE,
                                                    subset='training')
# Generate validation data from images
validate_generator = train_datagen.flow_from_dataframe(puery_results,
                                                       x_col='filename',
                                                       y_col='class',
                                                       target_size=(IMG_SIZE, IMG_SIZE, 3),
                                                       color_mode='rgb',
                                                       batch_size=BATCH_SIZE,
                                                       subset='validation')


# DEFINE AND COMPILE MODEL TODO: Copy final model design over from Colab
'''
# Import ResNet50 model from keras

# Model definition
cnn = keras.Sequential()
# Layer construction TODO: add L2 regularization on convolutional layers
cnn.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
cnn.add(keras.layers.MaxPooling2D((2, 2)))
cnn.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(keras.layers.MaxPooling2D((2, 2)))
cnn.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(32, activation='relu'))
cnn.add(keras.layers.Dense(len(class_names), activation='softmax'))

# Model compilation
cnn.compile(optimizer='RMSprop',
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['categorical_accuracy'])
cnn.summary()

# Model training TODO: incorporate early stopping callbacks
history = cnn.fit(train_generator,
                  steps_per_epoch=len(train_generator.filenames) // BATCH_SIZE,
                  epochs=16)
'''

# Export model for future use
# TODO: export model to pickle