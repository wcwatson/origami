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
    # TODO: SELECT the relevant things
    cur.close()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
        print('Database connection closed.')

# Segment into train and test sets

# Model initialization
# TODO: define CNN of some kind

# Model training
# TODO: train the model on the read-in data, use an automatic stop to prevent overfitting

# Export model for future use
# TODO: export model to an HDF5 file