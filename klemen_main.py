# Arguing agents start again
import pandas as pd
import numpy as np
import os,csv

path_to_data = os.getcwd() + '/Corpora/complete/'

files = os.listdir(path_to_data)
files = list(filter(lambda k: '.tsv' in k,files)) # keep only tsv files

lengths = []
temp_frames = []
df = pd.DataFrame()
for file in files:
	print("Reading {}".format(file))
	temp_df = pd.read_csv(path_to_data+file,delimiter='	',quoting=csv.QUOTE_NONE) # using tab delimiter (it may look like a space, but its a tab (probably))
	temp_frames.append(temp_df)
	lengths.append(len(temp_df))

df = pd.concat(temp_frames,ignore_index=True)

print(df[int(lengths[0]-5):int(lengths[0]+5)])