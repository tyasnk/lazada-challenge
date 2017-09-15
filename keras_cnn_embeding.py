# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:07:52 2017

@author: dwipr
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

#import lazada_utils
#%%
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.25
GLOVE_DIR = 'D:/Python Scripts/'

raw_train_dataset = pd.read_csv('data/training/data_train.csv', header=None)
raw_validation_dataset = pd.read_csv('data/validation/data_valid.csv', header=None)

train_clarity_y = np.loadtxt("data/training/clarity_train.labels", dtype=int)
train_conciseness_y = np.loadtxt("data/training/conciseness_train.labels", dtype=int)

# Use conciseness as label
labels = train_conciseness_y
#labels = train_clarity_y

titles= pd.DataFrame(pd.concat([pd.DataFrame(raw_train_dataset[2].values, columns=['title']), pd.DataFrame(train_conciseness_y, columns=['conciseness'])], axis=1))
texts = [i for i in titles['title']]


val_titles = pd.DataFrame(raw_validation_dataset[2].values, columns=['title'])
val_texts = [i for i in val_titles['title']]
#%%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

val_tokenizer = Tokenizer()
val_tokenizer.fit_on_texts(val_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)

val_word_index = val_tokenizer.word_index
print('Found %s unique val tokens.'%len(val_word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
val_data = pad_sequences(val_sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

#%%
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'),encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
#
print('Found %s word vectors.' % len(embeddings_index))

#%%
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
val_embedding_matrix = np.zeros((len(val_word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
#%%

for word, i in val_word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    # word not found in embedding index will be all-zeros.
    embedding_matrix[i] = embedding_vector
from keras.layers import Embedding
#
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


#%%
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, LSTM, GRU, BatchNormalization
from keras.optimizers import Adam


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
#%%
#x = Conv1D(32, 3, activation='relu',)(embedded_sequences)
#x = Conv1D(32, 3, activation='relu')(x)
#x = MaxPooling1D(3)(x)
#x = Dropout(0.5)(x)

#x = Conv1D(128, 3, activation='relu')(x)
#x = Conv1D(128, 3, activation='relu')(x)
#x = MaxPooling1D(3)(x)
#x = Dropout(0.5)(x)

#x = Conv1D(256, 7, activation='relu')(x)
#x = Conv1D(256, 7, activation='relu')(x)
#x = MaxPooling1D(7)(x)

#x = Flatten()(x)
#x = Dense(256, activation='relu')(x)
#x = Dropout(0.5)(x)
#x = Dense(128, activation='relu')(x)
#x = Dropout(0.5)(x)

x = GRU(units=16, dropout=0.5)(embedded_sequences)
#x = GRU(units=32, dropout=0.5)(x)
#x = (Dropout(0.3))(x)
x = BatchNormalization()(x)


preds = Dense(2, activation='softmax')(x)

adam = Adam(lr=0.001, clipvalue=0.5)
#rmsprop = RMSProp(lr=0.001)

model = Model(sequence_input, preds)

#model.load_weights("LSTM_64.h5")
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])
## happy learning!
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=27, batch_size=128,  )
model.save_weights("Clarity_GRU_32.h5")
#%%
#lazada_utils.plot_keras_history(hist)

pred_y = model.predict(val_data)
#%%
pd.DataFrame(pred_y).to_csv('pred_conciseness.csv', header=False)
#pd.DataFrame(pred_y).to_csv('pred_c.csv', header=False)