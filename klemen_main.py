# Arguing agents start again
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os,csv, time
import spacy # has pos tagger, word vector stuff and a bunch of other tools, very fast as well
from tqdm import tqdm
top_path = 'Corpora/complete/'
path_to_data = os.getcwd() + '/' + top_path

global_timer = time.time()

files = os.listdir(path_to_data)
files = list(filter(lambda k: '.tsv' in k,files)) # keep only tsv files

temp_frames = []
df = pd.DataFrame()
for file in files:
	print("Reading {}".format(file))
	temp_df = pd.read_csv(path_to_data+file,delimiter='	',quoting=csv.QUOTE_NONE) # using tab delimiter (it may look like a space, but its a tab (probably))
	# note on the 'quoting=csv.QUOTE_NONE': its there because in one of the files (school_uniforms.tsv) there's at least one entry that looks to the parser like an end of file character inside a string
	temp_frames.append(temp_df)

df = pd.concat(temp_frames,ignore_index=True)

# adding simpler labels
def labeling(row):
	if row['annotation'] == 'NoArgument':
		return 'NoArgument'
	elif row['annotation'] in ['Argument_against','Argument_for']:
		return 'Argument'
# apply labels
df['label'] = df.apply(lambda row: labeling(row),axis=1)

### Once we have the data, time to initialize the spacy parser
model = 'en_core_web_lg'
print("Loading model {}".format(model))
start_time = time.time()
nlp = spacy.load(model) # this is the large sized model that also supports word vectors (for some reason loads faster than medium size model)
print("Done. Loading took {} seconds".format((time.time()-start_time)))


# extract features from sentences (try different methods)
def feature_word_vector(sentence):
	# given a sentence, extracts word vectors
	sentence = nlp(sentence)
	features = pd.DataFrame()
	for w in sentence:
		if(not w.is_stop and not w.pos_ == "PUNCT"):
			features = features.append(pd.Series(w.vector),ignore_index=True)
	return features

def extract_features(row):
	features = feature_word_vector(row['sentence'])
	return features

# create a row with all the features
tqdm.pandas()
df['features'] = df.progress_apply(lambda row: extract_features(row),axis=1)
'''
for row in tqdm(range(2)):
	features = feature_word_vector(df.iloc[row]['sentence'])
	label = df.iloc[row]['label']
	#data = data.append(new_entry,ignore_index=True)
'''
df.info()
write_name1 = 'extracted_vectors1.json'
write_name2 = 'extracted_vectors2.json'
#write_path = top_path + write_name
print("Converting Data Frame 1 to JSON")
sub_frame1 = df[:12000]
out1 = sub_frame1.to_json()
print("Converting Data Frame 2 to JSON")
sub_frame2 = df[12000:]
out2 = sub_frame2.to_json()
print("Writing dataframe 1 with features to {}.".format(write_name1))
with open(write_name1,'w+') as f:
	f.write(out1)
print("Writing dataframe 2 with features to {}.".format(write_name2))
with open(write_name2,'w+') as h:
	h.write(out2)
print("Done.")
print("Total time elapsed: {}s".format((time.time()-global_timer)))


