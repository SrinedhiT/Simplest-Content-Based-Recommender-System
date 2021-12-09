from flask import Flask, render_template, request
import sys
sys.path.append('c:/users/lampe/appdata/local/programs/python/python38/lib/site-packages')
from rake_nltk import Rake

import nltk
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import count_matrix


app = Flask(__name__)

data=pd.read_json("TrialData.json")

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/submit',methods=['POST'])
def submit():
    if request.method == 'POST':
        club = request.form['club']
        club_name  = club
        recommended_club = count_matrix.recommend(club_name)
        L1 = recommended_club
        return render_template('results.html',result=recommended_club, C1 = L1[0], C2 = L1[1])

if __name__ == "__main__":
   app.run(debug=True)