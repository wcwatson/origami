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
import cv2
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

# Auxiliary function for image manipulation
def get_thresholds(img, window_size):
    m = np.median(img)
    low = int(max(0, (1.0 - window_size)*m))
    high = int(min(255, (1.0 + window_size)*m))
    return [low, high]

# Auxiliary function to process images
def preprocess_image(img):
    # Convert to grayscale
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)
    # Apply bilateral filtering to remove details of paper design
    img = cv2.bilateralFilter(img, 5, 50, 50)
    # Get contours using Canny filter
    WINDOW_SIZE = 0.2
    thresholds = get_thresholds(img, WINDOW_SIZE)
    img = np.float64(cv2.Canny(np.uint8(img), thresholds[0], thresholds[1]))
    # Repeat along three channels to make acceptable input for ResNet50
    img = np.array([img])
    img = img.transpose(1, 2, 0)
    img = tf.repeat(img, 3, axis=2)
    img = tf.expand_dims(img, axis=0)
    return img


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


# Page(s) to output classification and directions to user
@app.route('/result', methods=['GET', 'POST'])
def output_result():
    # Pull image from input field, temporarily save to static folder
    user_img = request.files['user_img']
    user_img.save(TEMP_IMG_FP) #TODO: make resilient to caching
    # Read in image, convert to expected input size for modeling
    IMG_SIZE = 128
    user_img = tf.keras.preprocessing.image.load_img(TEMP_IMG_FP, target_size=(IMG_SIZE, IMG_SIZE))
    user_img = tf.keras.preprocessing.image.img_to_array(user_img)
    # Convert to grayscale, check that the image is actually of origami
    user_img_gray = tf.expand_dims(tf.repeat(tf.image.rgb_to_grayscale(user_img), repeats=3, axis=2), 0)
    is_origami = ori_not_model.predict_classes(user_img_gray)[0]
    # If if is origami, proceed to normal results page
    if is_origami == 1:
        # Adjust image colors, predict class
        user_img_processed = preprocess_image(user_img)
        img_class = origami_classnames[origami_classifier.predict_classes(user_img_processed)[0]]
        # Set file path for instructions TODO: update to include links to sites and video
        instruction_fp = '../static/instructions/' + img_class + '/1.jpg'
        # Serve results page with appropriate instructions
        return render_template('result.html', img_class=img_class, instruction_fp=instruction_fp)
    # If the image is not origami, proceed to no_result page
    else:
        return render_template('no_result.html')