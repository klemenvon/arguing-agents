# Parser for auracaria dataset
import json
import os

path_to_corpus = os.getcwd() + '/Corpora/araucaria/'


class t_lPair():
	# object for a training/label pair
	def __init__(self,pairID):
		self.ID = pairID
		self.parse()

	def parse(self):
		# takes text and json files and parses them into two data points in this object
		# text first
		base_path = path_to_corpus + "nodeset" + str(self.ID)
		text_path = base_path + ".txt"
		self.plain = ""
		self.labeled = ""
		if os.path.isfile(text_path):
			with open(text_path,'r') as f:
				self.plain = f.read()
			# json file next, this one's a little more complex
			json_path = base_path + ".json"
			with open(json_path,'r') as f:
				self.labeled = json.load(f)
		