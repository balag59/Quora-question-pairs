
# coding: utf-8

# In[18]:


import pandas as pd


# In[19]:


data = pd.read_csv('data/train.csv')


# In[20]:


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


# In[21]:


maxlen = 40
batch_size = 64


# In[22]:


tok = text.Tokenizer(num_words=200000)
total_data = list(data['question1'].astype(str)) + list(data['question2'].astype(str))
tok.fit_on_texts(total_data)
x1 = tok.texts_to_sequences(data['question1'].astype(str))
x1 = sequence.pad_sequences(x1, maxlen=maxlen)

x2 = tok.texts_to_sequences(data['question2'].astype(str))
x2 = sequence.pad_sequences(x2, maxlen=maxlen)

word_index = tok.word_index


# In[ ]:


y_cat = to_categorical(data['is_duplicate'], num_classes=2)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


model1 = Sequential()
model2 = Sequential()


# In[ ]:


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


# In[ ]:


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


# In[ ]:


#mergind the two models of question 1 and question2
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


# In[ ]:


#model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


print(model.summary())


# In[ ]:


#checkpointer = ModelCheckpoint(filepath='model/model-{epoch:02d}.hdf5', verbose=1)


# In[ ]:


"""model.fit([x1,x2],y_cat,
          batch_size=batch_size,
          epochs=12,
          validation_split=0.1,
          callbacks=[checkpointer])"""


# In[ ]:


model = load_model('model/model-final.hdf5')


# In[ ]:


question1 = np.array(["What are some of the best romantic movies in English?"])
question2 = np.array(["What is the best romantic movie you have ever seen?"])
test_sentence1 = tok.texts_to_sequences(question1)
test_sentence1 = sequence.pad_sequences(test_sentence1, maxlen=maxlen)
test_sentence2 = tok.texts_to_sequences(question2)
test_sentence2 = sequence.pad_sequences(test_sentence2, maxlen=maxlen)
question_pair = [test_sentence1,test_sentence2]
labels = ['Not Duplicate','Duplicate']


# In[ ]:


pred = model.predict(question_pair)
for i in range(len(pred)):
    #print(pred[i])
    print("%s prediction with %.2f%% probability" % (labels[np.argmax(pred[i])], pred[i][np.argmax(pred[i])] * 100))
    #print("Duplicate prediction with %.2f%% probability" %pred[i][1] * 100)


# In[ ]:
