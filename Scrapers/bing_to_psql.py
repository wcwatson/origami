# Inserts information about image files into PostgreSQL; image files scraped using bing_images_download.py from Terminal

# Libraries
import psycopg2
from pathlib import Path
from PIL import Image
from io import BytesIO

# Global variables - sue me.
BASE_PATH = '/Users/wwatson/Desktop/Insight/Project/origami/Images/downloads/'
categories = ['butterfly', 'crane', 'duck', 'frog', 'star'] #TODO: make this dynamic so it knows classes from directories

# Function to write image information to origami database
def write_to_db(conn, cur, sql, image_path, image_category):
    # Write to database TODO: also read in image files themselves?
    cur.execute(sql, [image_path, image_category])
    conn.commit()
    print('Wrote type {} ({}) to database.'.format(image_category, image_path))


# FETCH IMAGES AND WRITE TO POSTGRES

# Connect to PostgreSQL database
try:
    print('Attempting to connect to PostgreSQL.')
    conn = psycopg2.connect(host='localhost', database='origami', user='postgres', password='postgres')
    cur = conn.cursor()
    sql = "INSERT INTO origami_images(image_path, image_class) VALUES((%s), (%s));"
    # Loop over category directories
    for cat in categories:
        cat_path = BASE_PATH + cat
        # Fetch all file paths in category directory
        img_pathlist = Path(cat_path).glob('*')
        for img_path in img_pathlist:
            # Convert to string
            img_path_str = str(img_path)
            print(img_path_str)
            write_to_db(conn, cur, sql, img_path_str, cat)
    cur.close()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
        print('PostgreSQL connection closed.')