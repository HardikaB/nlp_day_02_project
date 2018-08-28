# %load q03_LDA/build.py
# Default imports

from sklearn.decomposition import LatentDirichletAllocation
from greyatomlib.nlp_day_02_project.q02_count_vectorizer_for_LDA.build import q02_count_vectorizer_for_LDA


# Write your solution here :
def q03_LDA(path):
    tf,v2=q02_count_vectorizer_for_LDA(path)
    
    lda = LatentDirichletAllocation(n_components=20, max_iter=500,learning_method='batch',random_state=1)
    lda.fit(tf)
    return lda.components_


#path='data/sessions.csv'
#q03_LDA(path)

