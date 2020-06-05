# views script for FoldFinder app

# Imports
from flask import render_template, request
from FoldFinder import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import h5py
import pandas as pd
import psycopg2
import PIL

# Connect to PostgreSQL
user = 'wwatson'
host = 'localhost'
dbname = 'origami'
db = create_engine('postgres://%s%s/%s'%(user, host, dbname))
con = None
con = psycopg2.connect(database=dbname, user=user)

# Unpickle model, spin up for user input
origami_cnn = [] #TODO: figure out how this works

# Landing page TODO: eventually make this a nice thing with a button
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


# Page to output classification and directions to user TODO: integrate model
@app.route('/result')
def output_result():
    # Pull image from input field
    '''
    user_img = request.args.get('user_img')
    '''
    # Feed image through model
    # TODO: write this
    img_class = request.args.get('img_class') # TODO: everything
    # SELECT filepath for instructions FROM origami_instructions
    sql_query = "SELECT instruction_path FROM origami_instructions WHERE instruction_class='%s'" %img_class
    query_results = pd.read_sql_query(sql_query, con)
    instruction_fp = query_results.iloc[0]['instruction_path']
    # Fetch image TODO everything
    return render_template('result.html', img_class=img_class, instruction_fp=instruction_fp)
