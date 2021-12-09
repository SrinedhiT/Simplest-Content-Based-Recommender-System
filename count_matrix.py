import sys
sys.path.append('c:/users/lampe/appdata/local/programs/python/python38/lib/site-packages')
from rake_nltk import Rake

import nltk
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")



data=pd.read_json("TrialData.json")

indices = pd.Series(data['Club_name'])

data['Club_description'] = data['Club_description'].str.replace('[^\w\s]','')
data['Club_type_keywords'] = data['Club_type_keywords'].str.replace('[^\w\s]','')


data['Key_words'] = ''
r = Rake()

for index, row in data.iterrows():
    r.extract_keywords_from_text(row['Club_type_keywords'])
    key_words_dict_scores = r.get_word_degrees()
    row['Key_words'] = list(key_words_dict_scores.keys())


data['Club_type_keywords'] = data['Club_type_keywords'].map(lambda x: x.split(','))
data['Club_description'] = data['Club_description'].map(lambda x: x.split(','))


for index, row in data.iterrows():
    row['Club_type_keywords'] = [x.lower().replace(' ','') for x in row['Club_type_keywords']]
    
for index, row in data.iterrows():
    row['Club_description'] = [x.lower().replace(' ','') for x in row['Club_description']]


data['dump_words'] = ''
columns = ['Club_description', 'Key_words']



for index, row in data.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['dump_words'] = words


data['dump_words'] = data['dump_words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')


data = data[['Club_name','dump_words']]


count = CountVectorizer()
count_mat = count.fit_transform(data['dump_words'])
count_mat

cosine_sim = cosine_similarity(count_mat, count_mat)

def recommend(club_name, cosine_sim = cosine_sim):
    recommended_club = []
    idx = indices[indices == club_name].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_indices = list(score_series.iloc[1:3].index)
 
    for i in top_indices:
        recommended_club.append(list(data['Club_name'])[i])

    return(recommended_club)
