import os
import sys
from nltk.tokenize import word_tokenize

#num_lines = sum(1 for lin in open("USER_TWEETS.txt"))
#print(num_lines)
# file is 39,532,926 lines longs (contains that many tweets)
import re
import contractions
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

 
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    


def get_y_values(filename):
    user_y_values = {}
    f_output = open("./" + filename)
    for line in f_output:
        words= line.split()
        user_y_values[words[0]] = words[1]
    return user_y_values

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
links_re = "https:/*"
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

weblinks = re.compile(links_re)
count = 0
count_new = 0
#with open("3365.txt") as inp:
#    for line in inp:
#        for word in preprocess(line):
#            count+=1
y_vals = get_y_values("polarity_oc.txt")

path = '/Users/janvipalan/NLP-Project/users/6'
count = 0
for filename in os.listdir(path):
    f_tokens = open("/Users/janvipalan/NLP-Project/tokens/" + filename, "w+")
    f_tokens_lines = open("/Users/janvipalan/NLP-Project/tokens_lines/" + filename, "w+")
    mega_tweet = []
    filename = path+"/" + filename
    with open(filename) as inp:
        for line in inp:
            line = replace_contractions(line)
            line_list = []
            for word in preprocess(line):
                match = re.search(weblinks,word)
                if (not match and word.isalnum()):
                    mega_tweet.append(word)
                    line_list.append(word)
            line_list_str= ', '.join(line_list)
            f_tokens_lines.write("\n" + line_list_str)
    mega_tweet_string = ', '.join(mega_tweet)
    f_tokens.write(mega_tweet_string)
    f_tokens.close()
    f_tokens_lines.close()
    count+=1

print (count)
