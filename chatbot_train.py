# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 12:47:33 2022

@author: Maanas
"""

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


ids=[]
groups = []
tagged_qs= []
ignchars = ['?', '!']
data = open('intents.json').read()
intent = json.loads(data)

for i in intent['intents']:
    for question in i['patterns']:
        w=nltk.word_tokenize(question)
        ids.extend(w)
        tagged_qs.append((w,i['tag']))
        
        if i['tag'] not in groups:
            groups.append(i['tag'])

ids= [lemmatizer.lemmatize(w.lower()) for w in ids if w not in ignchars]
ids=sorted(list(set(ids)))
groups=sorted(list(set(groups)))
print(len(tagged_qs),'tagged questions: ')
print(tagged_qs)
print(len(groups),'tags')
print(groups)
print(len(ids),'different root words')
print(ids)

train=[]
empty = [0] * len(groups)
print(empty)

for j in tagged_qs:
    #j is a single tuple (question associated with tag)
    packet=[]
    patt=j[0]
    patt = [lemmatizer.lemmatize(m.lower()) for m in patt]
    for x in ids:
        packet.append(1) if x in patt else packet.append(0)
    #packet is a list of 0s and 1s, created for each query, 1 if a word is in the query and 0 if it isn't
    output_row=list(empty)
    output_row[groups.index(j[1])] = 1
    #output_row is a list row with 0s and 1s, each element signifying a tag. If the tagged_q belongs to a certain tag the value returned is 1, otherwise 0 is returned.
    train.append([packet,output_row])
    #train is an array
random.shuffle(train)
train=np.array(train)

tx=list(train[:,0])
#tx=groups
#ty=intent
ty=list(train[:,1])
#list of lists of 0s and 1s signifying if element is in a certain group/intent
print('Training data has been obtained')
print(tx)

#Creation of model 
model = Sequential()
model.add(Dense(128, input_shape=(len(tx[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(ty[0]), activation='softmax'))
model.summary()



# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(tx), np.array(ty), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)

print("model created")






















