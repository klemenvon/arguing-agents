# Start of the prject
# Data is in plain text and Json format
# Use Spacy for linguistic processing, pos tagging to add features and such

# TODO - build quick test parser of training data. Json specifically. DONE AND TESTED
# TODO - identify missing nodeset IDs because its fucking up the program (check file existance before reading in parse()?)

import os

from Parsers import araucaria_new

path_to_corpus = os.getcwd() + '/Corpora/araucaria/'

arConstObj = araucaria_new.ConstDataSet()

arConstObj.readFile("0", path_to_corpus)
arConstObj.readFile("1", path_to_corpus)

#print(len(data))
#print(data[0].plain)