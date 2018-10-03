#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 19:29:38 2018
"""

import os, json
from os.path import basename
import ntpath
import difflib
import pandas as pd
import nltk
import sklearn
from sklearn import svm
from nltk.tokenize import sent_tokenize
path_to_corpus = os.getcwd() + '/Corpora/araucaria/'


class ConstDataSet(object):
    def __init__(self):
        self.dataset = pd.DataFrame(columns=['Input', 'Output'])
        self.tagDataSet = pd.DataFrame(columns=['Input', 'Output'])
        self.training_datafr = pd.DataFrame(columns=['Input', 'Output'])
        self.switcher = {
                "0": self.readJsonFileAndConstructDataset,
                "1": self.readTxtFileAndConstructDataset,
                "2": self.readTsvFileAndConstructDataset
        }
    def readJsonFileAndConstructDataset(self,path_to_json):
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        json_data = pd.DataFrame(columns=['Input', 'Output'])
        for index, js in enumerate(json_files):
            with open(os.path.join(path_to_json, js)) as json_file:
                json_text = json.load(json_file)
                for arTextNode in json_text['nodes']:
                    if  arTextNode['text'] and (arTextNode['text'] != 'RA' and arTextNode['text'] != 'CA') and not(arTextNode['text'].startswith('YA_')):
                        json_data.loc[index] = [arTextNode['text'], '1']
        
        self.dataset = self.dataset.append(json_data)
        print(self.dataset.size)
        
    
        
    def readTxtFileAndConstructDataset(self,path_to_txt):
        non_arg_list = [];
        txt_files = [pos_txt for pos_txt in os.listdir(path_to_txt) if pos_txt.endswith('.txt')]
        jsons_data = pd.DataFrame(columns=['Input', 'Output'])
        for index, tx in enumerate(txt_files):
            with open(os.path.join(path_to_txt, tx)) as txt_file:
                if (ntpath.basename(txt_file.name)+"") != "araucaria-concatenated.txt":
                    text_data = txt_file.read()
                    sentences = sent_tokenize(text_data)
                    for sen in sentences:
                        non_arg_list.append(sen)
        self.parseNonArgData(non_arg_list)
                
    def parseNonArgData(self,non_arg_list):
        arg_list = self.dataset['Input']
        for each_non_arg in non_arg_list:
            max_match_sent = ""
            max_Match = 0;
            
            for arg in arg_list:
                SM = difflib.SequenceMatcher(None, each_non_arg, arg)
                match  = SM.find_longest_match(0, len(each_non_arg), 0, len(arg))
                if match.size > max_Match:
                    max_match_sent = arg;
                max_Match = max(match.size, max_Match)
            
            #if (len(each_non_arg)-max_Match <20):
                
            if (len(each_non_arg)-max_Match >20):
                eachNonArgDf = pd.DataFrame(columns=['Input', 'Output'])
                eachNonArgDf.loc[0] = [each_non_arg, '0']
                self.dataset.append(eachNonArgDf)
        
        self.dataset.to_csv('dataset.csv')
                
    def readTsvFileAndConstructDataset(self, path_to_tsv):
        tsv_files = [pos_tsv for pos_tsv in os.listdir(path_to_tsv) if pos_tsv.endswith('.tsv')]
        
        datafr_index=0;
        for index, tsv_each_path in enumerate(tsv_files):
            if tsv_each_path != "school_uniforms.tsv":
                print("Reading file "+tsv_each_path)
                train_file = pd.read_csv(os.path.join(path_to_tsv, tsv_each_path), delimiter='\t',encoding='utf-8', error_bad_lines=False)
                for row_index, each_train in train_file.iterrows():
                    if each_train[5] == 'NoArgument':
                        self.training_datafr.loc[datafr_index] = [each_train[4], '0']
                    else :
                        self.training_datafr.loc[datafr_index] = [each_train[4], '1']
                    datafr_index += 1
#                print(list(train_file.columns.values)) #file header
#                print(train_file.tail(35))
                #last N rows
        
        print(self.training_datafr.head(30))
                
    def extractVerbs(self):
        for index, row in self.training_datafr.iterrows():
            sen_tok = nltk.word_tokenize(row['Input'])
            sen_tags = nltk.pos_tag(sen_tok)
            tag_list = []
            for word, tag in sen_tags:
                if tag == 'VB' or tag == 'VBP' or tag == 'RB':
                    tag_list.append(word)
            
            self.tagDataSet.loc[index] = [tag_list, row['Output']]
        
        print(self.tagDataSet.head(30))
                    
                
    def buildSvm(self):
        print("Building SVM")
        svm_model = svm.SVC(gamma='scale')
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split( self.tagDataSet['Input'], self.tagDataSet['Output'], test_size=0.33, random_state=42)
        print(X_train)
        print(X_test)
        svm_model.fit(X_train, X_test)
        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        svm_model.predict(X_test, y_test)
        
        
        
        
        
        
        
    def readFile(self, type, path):
        # Get the function from switcher dictionary
        func = self.switcher.get(type)
        # Execute the function
        func(path)
        
    
        
        
        
        
        
