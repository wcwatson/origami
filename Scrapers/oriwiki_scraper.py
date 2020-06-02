# Simple scraper to collect some origami images and labels from oriwiki.com

# Libraries
import requests
from bs4 import BeautifulSoup
import time

from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd


# OrigamImage class: for temporary storage of data
class OrigamImage:
    def __init__(self, id, img_url, category):
        self.id = id
        self.img_url = img_url
        self.category = category

    def summ(self):
        print('ID: ' + self.id + ' | cat: ' + self.category + ' | img_url: ' + self.img_url)


# FETCH RESULTS FROM SEARCH PAGE, STORE IN TEMP STRUCT

# Search parameters and temp data storage
base_url = 'https://www.oriwiki.com/'
image_categories = ['butterfly', 'crane', 'duck', 'frog', 'star']
images = []

# Loop through image categories, get search results for each
for cat in image_categories:
    cat_url = base_url + 'searchresultsModels.php?IC=&ExactWord=&Term1=' + cat + '&Term2=&Term3=&Models=1'
    # Get search results
    cat_response = requests.get(cat_url)
    time.sleep(0.2)
    if cat_response:
        cat_soup = BeautifulSoup(cat_response.content, 'html.parser')
        # Get model IDs relevant to search parameters by searching through results table
        cat_rows = cat_soup.find('td', class_='section').find_all('tr')
        for row in cat_rows:
            cells = row.find_all('td')
            # Filter for header row
            if len(cells) > 1:
                for cell in cells:
                    model_id = '0'
                    image_url = 'images/NoModelImage.jpg'
                    # Get model ID
                    if cell.find('a'):
                        model_id = cell.find('a')['href'][22:]
                    # Get image URL
                    if cell.find('img'):
                        image_url = cell.find('img')['src']
                    # Create object and add to images
                    images.append(OrigamImage(model_id, image_url, cat))

# Check working
''''''
for item in images:
    item.summ()
''''''

# FETCH IMAGES AND WRITE TO POSTGRES

# PostgreSQL cursor
conn = psycopg2.connect(host='localhost', database='origami', user='postgres', password='postgres')
cur = conn.cursor()

# Loop over images
for image in images:
    # Process image URL
    image_url = image.img_url
    # Ignore NoModelImage.jpg, modify to deal with internally hosted images
    if image_url == 'images/NoModelImage.jpg':
        image_url = 'NONE'
    if image_url[:7] == 'images/':
        image_url = base_url + image_url
    # Fetch image file if it is meaningful
    if image_url != 'NONE':
        image_response = requests.get(image_url)
        time.sleep(0.2)
        if image_response:
            image_bytes = BytesIO(image_response.content)
            image_file = Image.open(image_bytes)
            #image_file.show()


