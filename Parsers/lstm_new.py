#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:46:13 2018

@author: sandy
"""

# imports needed and logging
import pandas as pd
#path_to_corpus = os.getcwd() + '/Corpora/araucaria/'
import gzip
import os
import gensim
from gensim.models import Word2Vec
import logging
import pandas as pd
#class ConstWord2Vec(object):
#    def __init__(self):
#         self.training_datafr = pd.DataFrame(columns=['Input', 'Output'])
         
#    def readTsvFileAndConstructDataset(self,path_to_tsv):
#        print("inside reading tsv ")
#        tsv_files = [pos_tsv for pos_tsv in os.listdir(path_to_tsv) if pos_tsv.endswith('.tsv')]
#        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#        datafr_index=0;
#        for index, tsv_each_path in enumerate(tsv_files):
#            print(tsv_files)
#            if tsv_each_path != "school_uniforms.tsv":
#                print("Reading file "+tsv_each_path)
#                train_file = pd.read_csv(os.path.join(path_to_tsv, tsv_each_path), delimiter='\t',encoding='utf-8', error_bad_lines=False)
#                for row_index, each_train in train_file.iterrows():
#                    if each_train[5] == 'NoArgument':
#                        self.training_datafr.loc[datafr_index] = [each_train[4], '0']
#                    else :
#                        self.training_datafr.loc[datafr_index] = [each_train[4], '1']
#                    yield gensim.utils.simple_preprocess(each_ttrain[4])
#                    datafr_index+=1
#        # build vocabulary and train model
#        model = gensim.models.Word2Vec(documents,
#        size=150,
#        window=10,
#        min_count=2,
#        workers=10)
#        model.train(documents, total_examples=len(documents), epochs=10)
#        print(self.training_datafr.head(2))
#        #shuffle the data
#        self.training_datafr = self.training_datafr.sample(frac=1).reset_index(drop=True)
#        print("--------------------")
#        print(len(self.training_datafr.loc[self.training_datafr['Output'] == '1']))
#        print(len(self.training_datafr.loc[self.training_datafr['Output'] == '0']))
#        print("--------------------")
#        print(self.training_datafr.head(2))

class ConstWord2Vec(object):
    def __init__(self):
        self.training_datafr = pd.DataFrame(columns=['Input', 'Output'])
        self.word_lists = []
        print("test new")
    def readTsvFileAndConstructDataset(self, path_to_tsv):
        print("tsv fileee")
        tsv_files = [pos_tsv for pos_tsv in os.listdir(path_to_tsv) if pos_tsv.endswith('.tsv')]
        datafr_index=0;
        for index, tsv_each_path in enumerate(tsv_files):
            print(tsv_files)
            if tsv_each_path != "school_uniforms.tsv":
                print("Reading file "+tsv_each_path)
                train_file = pd.read_csv(os.path.join(path_to_tsv, tsv_each_path), delimiter='\t',encoding='utf-8', error_bad_lines=False)
                for row_index, each_train in train_file.iterrows():
                    if each_train[5] == 'NoArgument':
                        self.training_datafr.loc[datafr_index] = [each_train[4], '0']
                    else :
                        self.training_datafr.loc[datafr_index] = [each_train[4], '1']
                    self.word_lists.append(gensim.utils.simple_preprocess(each_train[4]))
                    datafr_index+=1
        
#        print(self.word_lists)
        model = gensim.models.Word2Vec(self.word_lists,size=150,window=3,min_count=10,workers=10)
        model.train(self.word_lists, total_examples=len(self.word_lists), epochs = 10)
        print("thee ")
        print(model.wv['the'])
        