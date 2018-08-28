# %load q02_count_vectorizer_for_LDA/build.py
# Default imports

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.nlp_day_02_project.q01_load_data_tfidf.build import q01_load_data_tfidf

from sklearn.feature_extraction.text import CountVectorizer

# Write your solution here:
def q02_count_vectorizer_for_LDA(path):
    df,tf_talktitle,tf_feat=q01_load_data_tfidf(path,0.5,2,1000)
    #X_train,X_test,y_train,y_test = train_test_split(df[:,:-1],df[:,-1],test_size = 0.3)
    c_vect=CountVectorizer(analyzer='word',ngram_range=(1, 1),min_df=0,stop_words='english')
    variable1=c_vect.fit_transform(df['talkTitle'])
    variable2=c_vect.get_feature_names()
    return variable1,variable2
#path='data/sessions.csv'
#q02_count_vectorizer_for_LDA(path)

