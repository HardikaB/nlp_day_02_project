# LDA(LatentDirichletAllocation)

This assignment comprises implementation of one of the advanced techniques of
natural language processing called as LDA(LatentDirichletAllocation)
which is used in topic modelling


## Write a function `q03_LDA` that :
- Makes use of output data from `q02_count_vectorizer_for_LDA` question.
- Fits the lda model to output matrix of the previous question.




### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- | 
| path | String | compulsory |  | Path of data folder |




### Returns:

| Return | dtype | description |
| --- | --- | --- | 
| variable1 | numpy.ndarray | Array of component matrix |

Note : While using `CountVectorizer` use the following parameters
- n_components=20
- random_state=1
- learning_method='batch'
- max_iter=500
