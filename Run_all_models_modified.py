from __future__ import print_function

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from time import time
from optparse import OptionParser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

class Bunch(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# output logs to stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# show how to call hashing function
op = OptionParser()
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


#============================================================================
# Read custom data
#============================================================================


polarity = []
user_tweets = []
balance_classes = 1 # if 0, it only trims class 4, else trims all classes to size cls_num_ext
cls_0_num = 0
cls_1_num = 0
cls_2_num = 0
cls_3_num = 0
cls_4_num = 0
cls_num_ext = 500
num_ext = 0
with open("Categorized_User_Polarity.txt") as inp:
    for line in inp:
        values    = line.split("\t") # id, polarity
        user_id   = values[0]
        user_file = "tokens_lines_test/" + user_id
        if os.path.isfile(user_file): # not all user IDs had tweets in the master tweet file
            if not balance_classes:
                if int(values[1]) == 4:
                    if num_ext < 2000:
                        polarity.append(int(values[1])) # save the polarity
                        with open(user_file, 'r') as inp: # save the
                            user_tweets.append(inp.read())
                    num_ext += 1
                else:
                    polarity.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        user_tweets.append(inp.read())
            else:
                if int(values[1]) == 0:
                    if cls_0_num < cls_num_ext:
                        polarity.append(int(values[1])) # save the polarity
                        with open(user_file, 'r') as inp: # save the
                            user_tweets.append(inp.read())
                    cls_0_num += 1
                if int(values[1]) == 0:
                    if cls_1_num < cls_num_ext:
                        polarity.append(int(values[1])) # save the polarity
                        with open(user_file, 'r') as inp: # save the
                            user_tweets.append(inp.read())
                    cls_1_num += 1
                if int(values[1]) == 0:
                    if cls_2_num < cls_num_ext:
                        polarity.append(int(values[1])) # save the polarity
                        with open(user_file, 'r') as inp: # save the
                            user_tweets.append(inp.read())
                    cls_2_num += 1
                if int(values[1]) == 0:
                    if cls_3_num < cls_num_ext:
                        polarity.append(int(values[1])) # save the polarity
                        with open(user_file, 'r') as inp: # save the
                            user_tweets.append(inp.read())
                    cls_3_num += 1
                else: # class 4
                    if cls_4_num < cls_num_ext:
                        polarity.append(int(values[1])) # save the polarity
                        with open(user_file, 'r') as inp: # save the
                            user_tweets.append(inp.read())
                    cls_4_num += 1



#============================================================================
# Split into training and testing sets
#============================================================================


#                                                                    X            y         % data used for testing
raw_X_train, raw_X_test, raw_y_train, raw_y_test = train_test_split(user_tweets, polarity, test_size=0.2)

categories              = ['far_left','mid_left','neutral','mid_right','far_right'] # for 5 classes
data_train              = Bunch()
data_train.data         = raw_X_train
data_train.target_names = ['far_left','mid_left','neutral','mid_right','far_right'] # for 5 classes
data_train.target       = raw_y_train
data_test               = Bunch()
data_test.data          = raw_X_test
data_test.target        = raw_y_test
target_names            = data_train.target_names # Note: order of labels in `target_names` can be different from `categories`

print('data loaded')


#============================================================================
# Create target vectors
#============================================================================

y_train, y_test = data_train.target, data_test.target
init_time = time()
if opts.use_hashing:
    print("Using hashing vectorizer")
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False, n_features=opts.n_features)
    X_train    = vectorizer.transform(data_train.data)
else:
    print("Using tfidf vectorizer")
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english') 
    X_train    = vectorizer.fit_transform(data_train.data)

print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test set")
init_time     = time()
X_test = vectorizer.transform(data_test.data)
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from word to a string of tokens
if opts.use_hashing:
    names_of_features = None
else:
    names_of_features = vectorizer.get_feature_names()

if names_of_features:
    names_of_features = np.asarray(names_of_features)

def trim(s):
    return s if len(s) <= 80 else s[:77] + "..."


#============================================================================
# Benchmark classifiers
#============================================================================


def benchmark(clf):
    print('_' * 80)
    print("Starting training: ")
    print(clf)
    init_time = time()
    clf.fit(X_train, y_train)
    time_to_train = time() - init_time
    print("train time: %0.3fs" % time_to_train)
    print("Starting testing: ")
    init_time    = time()
    prediction   = clf.predict(X_test)
    time_to_test = time() - init_time
    print("test time:  %0.3fs" % time_to_test)
    print("Scoring the model: ")
    accuracy = metrics.accuracy_score(y_test, prediction)
    print("accuracy:   %0.3f" % accuracy)
    clf_descr = str(clf).split('(')[0]
    return clf_descr, accuracy, time_to_train, time_to_test

all_model_results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50, tol=1e-3), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    all_model_results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    all_model_results.append(benchmark(LinearSVC(penalty=penalty, dual=False, tol=1e-3))) # Create and train lib-linear models
    all_model_results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50, penalty=penalty))) # Create and train stochastic gradient models

print('=' * 80)
print("SGD with Elastic-Net")
all_model_results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50, penalty="elasticnet"))) # Create and train SGD w/ elastic penalty

print('=' * 80)
print("Nearest Centroid")
all_model_results.append(benchmark(NearestCentroid())) # Train NearestCentroid without threshold

print('=' * 80)
print("Naive Bayes (multinomial, bernoulli, and complement)")
all_model_results.append(benchmark(MultinomialNB(alpha=.01))) # Train sparse Naive Bayes classifiers
all_model_results.append(benchmark(BernoulliNB(alpha=.01)))
all_model_results.append(benchmark(ComplementNB(alpha=.1)))

print('=' * 80)
print("Linear SVC with l1")
all_model_results.append(benchmark(Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
                                   ('classification', LinearSVC(penalty="l2"))])))

  
# ==============================
# Plotting Accuracies Of All Models
# ==============================

idx = np.arange(len(all_model_results))
all_model_results = [[x[i] for x in all_model_results] for i in range(4)]

all_clfs, accuracy, time_for_training, time_to_test = all_model_results
time_for_training = np.array(time_for_training) / np.max(time_for_training)
time_to_test      = np.array(time_to_test) / np.max(time_to_test)

plt.figure(figsize=(10, 10))
plt.title("Classifier Accuracy")
plt.barh(idx, accuracy, .2, label="Accuracy", color='navy')
plt.barh(idx + .3, time_for_training, .2, label="Training Time", color='red')
plt.barh(idx + .6, time_to_test, .2, label="Test Time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(idx, all_clfs):
    plt.text(-.3, i, c)

plt.show()
