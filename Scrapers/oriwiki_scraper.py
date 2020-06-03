# Simple scraper to collect some origami images and labels from oriwiki.com


# Libraries
import requests
from bs4 import BeautifulSoup
import time
import psycopg2
from PIL import Image
from io import BytesIO

# Global variable - sue me.
BASE_URL = 'https://www.oriwiki.com/'


# OrigamImage class: for temporary storage of data
class OrigamImage:
    def __init__(self, id, img_url, category):
        self.id = id
        self.img_url = img_url
        self.category = category

    def summ(self):
        print('ID: ' + self.id + ' | cat: ' + self.category + ' | img_url: ' + self.img_url)


# Function to format results in one cell
def format_cell(cell, cat):
    model_id = '0'
    image_url = 'images/NoModelImage.jpg'
    # Get model ID
    if cell.find('a'):
        model_id = cell.find('a')['href'][22:]
    # Get image URL
    if cell.find('img'):
        image_url = cell.find('img')['src']
    # Create and return object
    return OrigamImage(model_id, image_url, cat)


# Function to fetch search results and insert into data structure
def scrape_oriwiki_results(cat):
    cat_url = BASE_URL + 'searchresultsModels.php?IC=&ExactWord=&Term1=' + cat + '&Term2=&Term3=&Models=1'
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
                # Append data in each cell to list
                for cell in cells:
                    images.append(format_cell(cell, cat))


# Function to process an image url as stored in temporary struct
def process_url(url):
    # Ignore NoModelImage.jpg
    if url == 'images/NoModelImage.jpg':
        url = 'NONE'
    # Modify to deal with internally hosted images
    if url[:7] == 'images/':
        url = BASE_URL + url
    return url


# Function to write image information to origami database (image guaranteed to be meaningful)
def write_to_db(conn, cur, sql, image):
    image_cat = image.category
    # Process image URL
    image_url = process_url(image.img_url)
    # Get image from URL
    image_response = requests.get(image_url)
    time.sleep(0.2)
    if image_response:
        image_bytes = BytesIO(image_response.content)
        image_file = Image.open(image_bytes)
        image_file.show()
        # Write to database TODO: fix so that it reads image files
        cur.execute(sql, [image_url, image_cat, image_file])
        #conn.commit()
    print('Wrote image {} ({}) to database.'.format(image.id, image.category))



# FETCH RESULTS FROM SEARCH PAGE, STORE IN TEMP STRUCT

# Set search parameters and temp data storage
image_categories = ['butterfly', 'crane', 'duck', 'frog', 'star']
images = []
# Loop through image categories, get search results for each
for cat in image_categories:
    scrape_oriwiki_results(cat)
# Check for functionality
'''
for item in images:
    item.summ()
'''
print(len(images))


# FETCH IMAGES AND WRITE TO POSTGRES

# Connect to PostgreSQL database
try:
    print('Attempting to connect to PostgreSQL...')
    conn = psycopg2.connect(host='localhost', database='origami', user='postgres', password='postgres')
    cur = conn.cursor()
    sql = "INSERT INTO origami(image_url, image_class, image_file) VALUES ((%image_url), (%image_cat), '(%image_file)');"
    # Loop over images
    for image in images:
        # If a meaningful image exists, write to db
        image_url = process_url(image.img_url)
        if image_url != 'NONE':
            write_to_db(conn, cur, sql, image)
    cur.close()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
        print('PostgreSQL connection closed.')