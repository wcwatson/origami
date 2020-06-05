# __init__ file for FoldFinder app

from flask import Flask
app = Flask(__name__)
from FoldFinder import views