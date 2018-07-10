import pandas as pd
data = pd.read_csv('data/train.csv')
import numpy as np
from keras.preprocessing import sequence,text
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, Activation, GRU, Input
from keras.layers.merge import concatenate
from keras.datasets import imdb
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
np.random.seed(7)

maxlen = 40
batch_size = 64

tok = text.Tokenizer(num_words=200000)
total_data = list(data['question1'].astype(str)) + list(data['question2'].astype(str))
tok.fit_on_texts(total_data)
x1 = tok.texts_to_sequences(data['question1'].astype(str))
x1 = sequence.pad_sequences(x1, maxlen=maxlen)
x2 = tok.texts_to_sequences(data['question2'].astype(str))
x2 = sequence.pad_sequences(x2, maxlen=maxlen)
word_index = tok.word_index
y_cat = to_categorical(data['is_duplicate'], num_classes=2)

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

#create the embedding matrix mapping every index in the corpus to it's respective embedding_vector
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model1 = Sequential()
model2 = Sequential()

model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=maxlen,
                     trainable=False))
model1.add(Bidirectional(LSTM(64,return_sequences=True)))
model1.add(Dropout(0.6))
model1.add(Bidirectional(GRU(64,return_sequences=True)))
model1.add(Dropout(0.6))
model1.add(Bidirectional(LSTM(64)))
model1.add(Dropout(0.6))

model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=maxlen,
                     trainable=False))
model2.add(Bidirectional(LSTM(64,return_sequences=True)))
model2.add(Dropout(0.6))
model2.add(Bidirectional(GRU(64,return_sequences=True)))
model2.add(Dropout(0.6))
model2.add(Bidirectional(LSTM(64)))
model2.add(Dropout(0.6))

#merging the two models of question 1 and question2
merged_model_output = concatenate([model1.output, model2.output])
merged_model_output = BatchNormalization()(merged_model_output)
merged_model_output = Dropout(0.5)(merged_model_output)
merged_model_output = Dense(128,activation='relu')(merged_model_output)
merged_model_output = BatchNormalization()(merged_model_output)
merged_model_output = Dropout(0.5)(merged_model_output)
merged_model_output = Dense(32,activation='relu')(merged_model_output)
merged_model_output = BatchNormalization()(merged_model_output)
merged_model_output = Dropout(0.5)(merged_model_output)
merged_model_output = Dense(2,activation='softmax')(merged_model_output)
model = Model([model1.input,model2.input],merged_model_output)

model = load_model('model/model-final.hdf5')

question1 = np.array(["What are some of the best romantic movies in English?"])
question2 = np.array(["What is the best romantic movie you have ever seen?"])
test_sentence1 = tok.texts_to_sequences(question1)
test_sentence1 = sequence.pad_sequences(test_sentence1, maxlen=maxlen)
test_sentence2 = tok.texts_to_sequences(question2)
test_sentence2 = sequence.pad_sequences(test_sentence2, maxlen=maxlen)
question_pair = [test_sentence1,test_sentence2]
labels = ['Not Duplicate','Duplicate']

pred = model.predict(question_pair)
for i in range(len(pred)):
    print("%s prediction with %.2f%% probability" % (labels[np.argmax(pred[i])], pred[i][np.argmax(pred[i])] * 100))
