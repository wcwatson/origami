# Scraper for Google Images...why wasn't this the first thing I did?
# DAMMIT, google_images_download no longer works as of February 2020

# Libraries
from google_images_download import google_images_download
import requests
from bs4 import BeautifulSoup
import time
import psycopg2
from PIL import Image
from io import BytesIO


# Set constant and global variables - sue me.
IMAGE_CATEGORIES = ['butterfly', 'crane', 'duck', 'frog', 'star'] #TODO: make this read in from a file?

# Initialize google scraper, set arguments
response = google_images_download.googleimagesdownload()
limit = 100 #TODO: Expand
color_type = 'black-and-white'
for cat in IMAGE_CATEGORIES:
    cat = 'origami ' + cat
    arguments = {'keywords':cat, 'limit':limit, 'color_type':color_type, 'print_urls':True}
    paths = response.download(arguments)
