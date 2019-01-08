import gzip
import gensim 
import os	
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_input(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            yield gensim.utils.simple_preprocess(line)

data_file = "user_tweets_no_stopwords.txt"

documents = list (read_input (data_file))
logging.info ("The data file has been read completely")

model = gensim.models.Word2Vec(documents, size=200, window=10, min_count=3, workers=5)
model.train(documents,total_examples=len(documents),epochs=10)
model.save("test_all.model")
