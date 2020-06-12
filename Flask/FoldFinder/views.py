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
import tensorflow as tf
import pickle
import h5py
import numpy as np
import pandas as pd
import psycopg2
from PIL import Image
import random

# Connect to PostgreSQL
user = 'wwatson'
host = 'localhost'
dbname = 'origami'
db = create_engine('postgres://%s%s/%s'%(user, host, dbname))
con = None
con = psycopg2.connect(database=dbname, user=user)

# Spin up models, prepare for user input
TEMP_IMG_FP = './FoldFinder/static/user_img.jpg'
ori_not_model = tf.keras.models.load_model('../Classifiers/ori_not/ori_not_model.h5')
origami_classifier = tf.keras.models.load_model('../Classifiers/origami_classifier/origami_classifier.h5')
origami_classnames = pickle.load(open('../Classifiers/origami_classifier/origami_classnames.p', 'rb'))


# APP FUNCTIONALITY

# Landing page
@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if os.path.exists(TEMP_IMG_FP):
        os.remove(TEMP_IMG_FP)
    return render_template('index.html')


# Page giving some information about the ethics of distributing origami designs.
@app.route('/ethics')
def ethics():
    return render_template('ethics.html')


# Page(s) to output classification and directions to user TODO: integrate models
@app.route('/result', methods=['GET', 'POST'])
def output_result():
    # Pull image from input field, temporarily save to static folder
    user_img = request.files['user_img']
    user_img.save(TEMP_IMG_FP)
    # Read in image, convert to expected input size for modeling
    IMG_SIZE = 128
    user_img = tf.keras.preprocessing.image.load_img(TEMP_IMG_FP, target_size=(IMG_SIZE, IMG_SIZE))
    user_img = tf.keras.preprocessing.image.img_to_array(user_img)
    # Convert to grayscale, check that the image is actually of origami
    user_img_gray = tf.expand_dims(tf.repeat(tf.image.rgb_to_grayscale(user_img), repeats=3, axis=2), 0)
    is_origami = ori_not_model.predict_classes(user_img_gray)[0]
    # If if is origami, proceed to normal results page
    if is_origami == 1:
        # Adjust image colors, predict class TODO: double-check saturation adjustment multipliers
        user_img_saturated = tf.expand_dims(tf.image.adjust_saturation(user_img, saturation_factor=2.5), 0)
        img_class = origami_classnames[origami_classifier.predict_classes(user_img_saturated)[0]]
        ''' NICE IDEA, BUT NO
        # Fetch appropriate instructions from database
        sql_query = "SELECT instruction_path FROM origami_instructions WHERE instruction_class = '%s'"%img_class
        query_result = pd.read_sql_query(sql_query, con)
        instruction_index = 0 # TODO: random.randint(0, len(query_result))?
        instruction_fp = query_result.iloc[instruction_index]['instruction_path'] #TODO: adapt for URL if needed
        '''
        instruction_fp = '../static/instructions/' + img_class + '/1.jpg'
        # Serve results page with appropriate instructions
        return render_template('result.html', img_class=img_class, instruction_fp=instruction_fp)
    # If the image is not origami, proceed to no_result page
    else:
        return render_template('no_result.html')