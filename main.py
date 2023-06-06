from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
#dashboard.bind(app)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    pass

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    pass

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    pass