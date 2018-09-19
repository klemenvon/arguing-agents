# Parser for auracaria dataset
import json
import os

path_to_corpus = 'Corpora/araucaria/'

if __name__ == "__main__":
	# if this is run as the main file, the path to corpus is different than when its run from the main file
	path_to_corpus = '../Corpora/araucaria'

class t_lPair():
	# object for a training/label pair
	def __init__(self,pairID):
		self.ID = pairID
		self.parse()

	def parse(self):
		base_path = path_to_corpus + "nodeset" + str(self.ID)
		# storing plain text sentences in a string
		text_path = base_path + ".txt"
		with open(text_path,'r') as f:
			self.plain = f.read()
		# json file next, this may require additional parsing to simplify later
		json_path = base_path + ".json"
		with open(json_path,'r') as f: # don't use 'rb' as a read option, the json parser can't handle that
			self.labeled = json.load(f)
