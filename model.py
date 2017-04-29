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

data = pd.read_csv('data/train.csv', sep=',')
y = data.is_duplicate.values
tok = text.Tokenizer(num_words=200000)
max_len = 40
tok.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tok.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tok.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

word_index = tok.word_index

ytrain_enc = np_utils.to_categorical(y)
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

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

merged_model = Sequential()
model1 = Sequential()
model2 = Sequential()

model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model1.add(LSTM(300,activation='softsign',return_sequences=True))
model1.add(GRU(300,activation='softsign',return_sequences=True))
model1.add(LSTM(300,activation='softsign'))

model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model2.add(LSTM(300,activation='softsign',return_sequences=True))
model2.add(GRU(300,activation='softsign',return_sequences=True))
model2.add(LSTM(300,activation='softsign'))

merged_model.add(Merge([model1, model2], mode='concat'))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)
merged_model.fit([x1, x2, x1, x2], y=y, batch_size=384,epochs=1,
                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])

scores = merged_model.evaluate([x1, x2, x1, x2],y)
print("\n%s: %.2f%%" % (merged_model.metrics_names[1], scores[1]*100))
