#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:46:13 2018

@author: sandy
"""

# imports needed and logging
import pandas as pd
import re
#path_to_corpus = os.getcwd() + '/Corpora/araucaria/'
import gzip
import os
os.environ['KERAS_BACKEND']='tensorflow'
import gensim
import sklearn
import nltk
import pickle
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import model_from_json
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras import utils
import json

import logging
import pandas as pd
import numpy as np

class ConstWord2Vec(object):
    def __init__(self):
        self.training_datafr = pd.DataFrame(columns=['Input', 'Output'])
        self.input = []
        self.output = []
        self.word_model = '';
        self.word_lists = []
        self.train_features = []
        self.output = []
        self.token_index = {}
    
    def readTsvFileAndConstructDataset(self, path_to_tsv):
        tsv_files = [pos_tsv for pos_tsv in os.listdir(path_to_tsv) if pos_tsv.endswith('.tsv')]
        csv_files = [pos_tsv for pos_tsv in os.listdir(path_to_tsv) if pos_tsv.endswith('.csv')]
        word_lem = WordNetLemmatizer()
        for index, tsv_each_path in enumerate(tsv_files):
            if tsv_each_path != "school_uniforms.tsv":
                print("Reading file "+tsv_each_path)
                train_file = pd.read_csv(os.path.join(path_to_tsv, tsv_each_path), delimiter='\t',encoding='utf-8', error_bad_lines=False)
                for row_index, each_train in train_file.iterrows():
                    regex_sen = re.sub('[^a-zA-z0-9\s]', '', each_train[4])
                    regex_sen = regex_sen.lower()
                    word_splits = word_tokenize(regex_sen)
                    word_splits = [word_lem.lemmatize(i, pos='v') for i in word_splits]
                    word_splits = [word_lem.lemmatize(i) for i in word_splits]
                    recon_sen = ' '.join(word for word in word_splits)
                    #considering only sentences having word length > 3
                    if len(word_splits) > 3:
                        self.word_lists.append(recon_sen)
                        if each_train[5] == 'NoArgument':
                            self.output.append('0')
                        else:
                            self.output.append('1') 
                    
        for index, csv_each_path in enumerate(csv_files):
            print(csv_files)
            train_file = pd.read_csv(os.path.join(path_to_tsv, csv_each_path), delimiter=',',encoding='utf-8', error_bad_lines=False)
            print("reading file "+csv_each_path)
            for row_index, each_train in train_file.iterrows():
                regex_sen = re.sub('[^a-zA-z0-9\s]', '', each_train[4])
                regex_sen = regex_sen.lower()
                word_splits = word_tokenize(regex_sen)
                word_splits = [word_lem.lemmatize(i, pos='v') for i in word_splits]
                recon_sen = ' '.join(word[0] for word in word_splits)
                #considering only sentences having word length > 3
                if len(word_splits) > 3:
                    self.word_lists.append(recon_sen)
                    if each_train[5] == 'NoArgument':
                        self.output.append('0')
                    else:
                        self.output.append('1') 
        
        keras_Tokenizer = Tokenizer()
        keras_Tokenizer.fit_on_texts(self.word_lists)
        encoded_sens = keras_Tokenizer.texts_to_sequences(self.word_lists)
        print(encoded_sens)
        vocab_size = len(keras_Tokenizer.word_counts)+1
        max_length = 150
        batch_size = 32
#       encoded_sens = [one_hot(d, vocab_size) for d in self.word_lists]
#        print(encoded_sens)
        padded_sens = pad_sequences(encoded_sens, maxlen=max_length, padding='post')
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=max_length))
        model.add(LSTM(max_length, dropout_U = 0.2, dropout_W = 0.2))
        model.add(Dense(2,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        print(model.summary())
        self.output = pd.get_dummies(self.output).values
        X_train, X_valid, Y_train, Y_valid = train_test_split(padded_sens,self.output, test_size = 0.20, random_state = 36)
        #Here we train the Network.
        model.fit(X_train, Y_train, batch_size =batch_size, nb_epoch = 5,  verbose = 5)
        score,acc = model.evaluate(X_valid, Y_valid, verbose = 2, batch_size = batch_size)
        print("validation accuracy ",acc)
        model_json = model.to_json()
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(keras_Tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        print("Saved model to disk")
        
    def load_model(self, sen_list, output):
        # load json and create model
        max_length = 150
        batch_size = 32
        tokenizer = ''
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        loaded_model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        encoded_sens = tokenizer.texts_to_sequences(sen_list)
        print("after text to sequence")
        padded_sens = pad_sequences(encoded_sens, maxlen=max_length, padding='post')
        score,acc = loaded_model.evaluate(padded_sens, np.array(output), verbose = 2, batch_size = batch_size)
        print("validation accuracy ",acc)
        
        
        
        
        
        
        
        
        
        
        
       
        
        