# Start of the prject
# Data is in plain text and Json format
# Use Spacy for linguistic processing, pos tagging to add features and such

# TODO - build quick test parser of training data. Json specifically. DONE AND TESTED
# TODO - identify missing nodeset IDs because its fucking up the program (check file existance before reading in parse()?)

from Parsers import araucaria

data = []

for x in range(7,217):
	data.append(araucaria.t_lPair(x))

print(len(data))
print(data[0].plain)