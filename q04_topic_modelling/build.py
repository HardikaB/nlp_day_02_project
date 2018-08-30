# %load q04_topic_modelling/build.py
# Default imports

import numpy as np
from greyatomlib.nlp_day_02_project.q02_count_vectorizer_for_LDA.build import q02_count_vectorizer_for_LDA
from greyatomlib.nlp_day_02_project.q03_LDA.build import q03_LDA


#  Write your solution here :
def q04_topic_modelling(path,n_top_words=20):
    v1,v2=q02_count_vectorizer_for_LDA(path)
    v3=q03_LDA(path)
    message=list()
    for topic_idx, topic in enumerate(v3):
        topic='Topic '+str(topic_idx) +': '+' '.join([v2[i] for i in topic.argsort()[:-n_top_words :-1]])
        message.append(topic)
    return message

    


#path='data/sessions.csv'
#q04_topic_modelling(path,20)

