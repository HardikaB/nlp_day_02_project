# %load q01_load_data_tfidf/build.py
# Default imports

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Write your solution here :

def q01_load_data_tfidf(path,max_df=0.5,min_df=2,no_features=1000):
    df=pd.read_csv(path)
    tf_vect=TfidfVectorizer(max_df=max_df,min_df=min_df,max_features=no_features)
    tt_feature=tf_vect.fit_transform(df['talkTitle'])
    variable3=tf_vect.get_feature_names()
    return df,tt_feature,variable3


