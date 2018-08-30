# %load q01_load_data_tfidf/build.py
# Default imports

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Write your solution here :

def q01_load_data_tfidf(path,max_df=0.95,min_df=2,no_features=1000):
    variable1=pd.read_csv(path)
    tf_vect=TfidfVectorizer(stop_words='english',max_df=max_df,min_df=min_df,max_features=no_features)
    variable2=tf_vect.fit_transform(variable1['talkTitle'])
    variable3=tf_vect.get_feature_names()
    return variable1,variable2,variable3


