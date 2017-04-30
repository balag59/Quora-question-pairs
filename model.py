#import dependencies
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Merge
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from keras.models import load_model

#read input training data
data = pd.read_csv('data/train.csv', sep=',')
y = data.is_duplicate.values

#create indices for all words in the corpus
tok = text.Tokenizer(num_words=200000)
max_len = 40
tok.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tok.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tok.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

word_index = tok.word_index

ytrain_enc = np_utils.to_categorical(y)

#create a dictionary which stores embeddings for every word
embeddings_index = {}
f = open('data/glove.840B.300d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype='float32')
    except:
        pass
    embeddings_index[word] = coefs
f.close()

#create the embedding matrix mapping every index in the corpus to it's respective embedding_vector
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#creating keras models to represent network architecture
merged_model = Sequential()
model1 = Sequential()
model2 = Sequential()

#emedding layer for question 1
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
# LSTM for question 1
model1.add(LSTM(300,activation='softsign',return_sequences=True))
#GRU for question 1
model1.add(GRU(300,activation='softsign',return_sequences=True))
# another LSTM for question 1
model1.add(LSTM(300,activation='softsign'))

#emedding layer for question 2
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
# LSTM for question 2
model2.add(LSTM(300,activation='softsign',return_sequences=True))
#GRU for question 2
model2.add(GRU(300,activation='softsign',return_sequences=True))
# another LSTM for question 2
model2.add(LSTM(300,activation='softsign'))

#mergind the two models of question 1 and question2
merged_model.add(Merge([model1, model2], mode='concat'))
#adding a layer of batch normalization
merged_model.add(BatchNormalization())
#adding a dense layer
merged_model.add(Dense(300))
#adding a ReLU activation function layer
merged_model.add(PReLU())
#adding a dense layer to convert vectors to size 1
merged_model.add(Dense(1))

#final activation layer to make predictions
merged_model.add(Activation('sigmoid'))

#specifying hyperparameters for loss and optimization
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#training the model and specifying the required parameters
merged_model.fit([x1, x2], y=y, batch_size=384,epochs=7,
                 verbose=1, validation_split=0.1, shuffle=True)

#saving the model to make future predictions
merged_model.save('final_model.h5')

#evaluating accuracy of the model
scores = merged_model.evaluate([x1, x2],y)
print("\n%s: %.2f%%" % (merged_model.metrics_names[1], scores[1]*100))
