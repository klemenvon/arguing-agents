#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 19:29:38 2018
"""

import os, json
import pandas as pd
path_to_corpus = os.getcwd() + '/Corpora/araucaria/'


class ConstDataSet(object):
    def __init__(self):
        self.dataset = pd.DataFrame(columns=['Input', 'Output'])
        self.switcher = {
                "0": self.readJsonFileAndConstructDataset,
                "1": self.readTxtFileAndConstructDataset
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
        txt_files = [pos_txt for pos_txt in os.listdir(path_to_txt) if pos_txt.endswith('.txt')]
        jsons_data = pd.DataFrame(columns=['Input', 'Output'])
        for index, tx in enumerate(txt_files):
            with open(os.path.join(path_to_txt, tx)) as txt_file:
                text_data = txt_file.read()
                print(text_data)
                
                
    
    def readFile(self, type, path):
        # Get the function from switcher dictionary
        func = self.switcher.get(type)
        # Execute the function
        func(path)
        
        
        
        
        

