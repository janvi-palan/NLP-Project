from tabulate import tabulate
import os
# %matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from nltk.corpus import stopwords 
import gensim
from gensim.models import Word2Vec
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier


X, y = [], []
test = []


with open("Categorized_User_Polarity.txt") as inp:
    for line in inp:
        # print (line)
        values    = line.split("\t") # id, polarity
        user_id   = values[0]
        user_file = "tokens_lines_test/" + user_id
        if os.path.isfile(user_file): # not all user IDs had tweets in the master tweet file
            y.append(int(values[1])) # save the polarity
            with open(user_file, 'r') as inp: # save the
            	X.append(inp.read().split())
                

X, y = np.array(X), np.array(y)
print ("total examples %s" % len(y))


with open("glove.6B.100d.txt", "rb") as lines:
    wvec = {line.split()[0].decode(encoding): np.array(line.split()[1:],dtype=np.float32)
               for line in lines}

glove_small = {}
all_words = set(w for words in X for w in words)
with open("glove.6B.100d.txt", "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode(encoding)
        if (word in all_words):
            nums=np.array(parts[1:], dtype=np.float32)
            glove_small[word] = nums


# print(len(all_words))

# svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),("linear svc", LinearSVC(penalty="l2", dual=False,tol=1e-3))])



class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=200
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    
# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=200
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)

        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


svc_embedding = Pipeline([("tfidf_glove_vectorizer", TfidfEmbeddingVectorizer(glove_small)), ("linear svc with w2v", LinearSVC(penalty="l2", dual=False,
                                       tol=1e-3))])
sgd_embedding = Pipeline([("tfidf_glove_vectorizer", TfidfEmbeddingVectorizer(glove_small)), ("linear sgd with w2v", (SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet")))])
arrange_all_models = [
    ("sgd", sgd_embedding),
    ("svc_tfidf", svc_tfidf),
    ("svc_w2v_small", svc_embedding),
]


unsorted_accuracies = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in arrange_all_models]
accuracy = sorted(unsorted_accuracies, key=lambda x: -x[1])


print (tabulate(accuracy, floatfmt=".4f", headers=("model", 'accuracy')))