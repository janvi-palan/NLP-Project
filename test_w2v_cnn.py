from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.activations import softmax
import gensim
from gensim.models import Word2Vec

BASE_DIR = ''
GLOVE_DIR = ''

Max_Length  = 10000 #max length of sequence 
Max_Word_Count = 20000
Dimensions = 200 #embedding dimensions 
Val_Split = 0.2 #Fraction of data for validation  

print('Initiating text processing')

texts = []  # list of text samples
label_ind  = {}  #  #{LabelName: Num_ID}
labels = []  


cls_0_num       = 0
cls_1_num       = 0
cls_2_num       = 0
cls_3_num       = 0
cls_4_num       = 0
cls_num_ext     = 600

with open("Categorized_User_Polarity.txt") as inp:
    for line in inp:
        values    = line.split("\t") # id, polarity
        user_id   = values[0]
        user_file = "tokens_lines_test_1/" + user_id+".txt"
        if os.path.isfile(user_file): # not all user IDs had tweets in the master tweet file
            if int(values[1]) == 0:
                if cls_0_num < cls_num_ext:
                    labels.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        texts.append(inp.read())
                cls_0_num += 1
            if int(values[1]) == 0:
                if cls_1_num < cls_num_ext:
                    labels.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        texts.append(inp.read())
                cls_1_num += 1
            if int(values[1]) == 0:
                if cls_2_num < cls_num_ext:
                    labels.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        texts.append(inp.read())
                cls_2_num += 1
            if int(values[1]) == 0:
                if cls_3_num < cls_num_ext:
                    labels.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        texts.append(inp.read())
                cls_3_num += 1
            else: # class 4
                if cls_4_num < cls_num_ext:
                    labels.append(int(values[1])) # save the polarity
                    with open(user_file, 'r') as inp: # save the
                        texts.append(inp.read())
                cls_4_num += 1


print('Working on %s docs.' % len(texts))

model = gensim.models.Word2Vec.load("test_all.model")
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}


tokenizer = Tokenizer(num_words=Max_Word_Count)
tokenizer.fit_on_texts(texts)
seq = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(seq, maxlen=Max_Length)

labels = to_categorical(np.asarray(labels))


#Val-train split
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


num_words = min(Max_Word_Count, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, Dimensions))
for word, i in word_index.items():
    if i > Max_Word_Count:
        continue
    embedding_vector = model.wv[model.wv.index2word[i]]
  
    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,
                            Dimensions,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=Max_Length,
                            trainable=False)

print('Training model.')


sequence_input = Input(shape=(Max_Length,), dtype='int32')
embedded_sequences_input = embedding_layer(sequence_input)
print(np.shape(embedded_sequences))

x = Conv1D(128, 5, activation='relu')(embedded_sequences_input)
x1 = MaxPooling1D(5)(x)
x2 = Conv1D(128, 5, activation='relu')(x1)
x2 = MaxPooling1D(5)(x2)
x3 = Conv1D(128, 5, activation='relu')(x2)
x3 = GlobalMaxPooling1D()(x3)
x4 = Dense(128, activation='relu')(x3)
#print (np.shape(x4))
predicted = Dense(5, activation='softmax')(x)


model = Model(sequence_input, predicted)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
#rmsprop
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
validation_data=(x_val, y_val))
