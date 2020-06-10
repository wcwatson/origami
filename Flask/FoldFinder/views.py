# /views script for FoldFinder Flask app
#
# Last modified: 2020-06-DD


# APP WARM-UP; IMPORT LIBRARIES, ESTABLISH DATABASE CONNECTION, SPIN UP MODELS

# Libraries
from flask import render_template, request
from FoldFinder import app
import os
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import psycopg2
import PIL
import random

# Connect to PostgreSQL
user = 'wwatson'
host = 'localhost'
dbname = 'origami'
db = create_engine('postgres://%s%s/%s'%(user, host, dbname))
con = None
con = psycopg2.connect(database=dbname, user=user)

# Spin up models, prepare for user input
ori_not_model = tf.keras.models.load_model('../../Classifiers/ori_not/ori_not_model.h5')
origami_classifier = tf.keras.models.load_model('../../Classifiers/origami_classifier/origami_classifier.h5') #TODO: check file name


# APP FUNCTIONALITY

# Landing page
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


# Page(s) to output classification and directions to user TODO: integrate models
@app.route('/result', methods=['GET', 'POST'])
def output_result():
    # Pull image from input field, convert to expected input size
    user_img = request.files['user_img']
    IMG_SIZE = 64
    # TODO: check input size, currently ori_not expects 64x64 rgb, origami_classifier expects 128x128 (rgb)
    user_img_resized = tf.keras.preprocessing.image.load_img(user_img, target_size=(IMG_SIZE, IMG_SIZE))
    # Check that the image is actually of origami
    is_origami = ori_not_model.predict(user_img_resized)
    # If if is origami, proceed to normal results page
    if is_origami: #TODO: check and adapt for non-boolean output
        # Get the origami class
        img_class = origami_classifier.predict(user_img_resized) #TODO: check and adapt for nature of output
        # Fetch appropriate instructions from database
        sql_query = "SELECT instruction_path FROM origami_instructions WHERE instruction_class = '%s'"%img_class
        query_result = pd.read_sql_query(sql_query, con)
        instruction_index = random.randint(0, len(query_result))
        instruction_fp = query_result.iloc[instruction_index]['instruction_path'] #TODO: adapt for URL if needed
        # Serve results page with appropriate instructions
        return render_template('result.html', user_img=user_img, img_class=img_class, instruction_fp=instruction_fp)
    # If the image is not origami, proceed to no_result page
    else:
        return render_template('no_result.html', user_img=user_img)