import pandas as pd
import numpy as np

naics = pd.read_excel('Naics3 (label) taxonomy.xlsx')
buisness_cat = pd.read_excel('Business category taxonomy.xlsx')

def create_dict(df):
    return dict(zip(df['naics_label'], df['description']))

naics_dict = create_dict(naics)
# print(naics_dict)

def eliminate_multimple_spaces_and_newlines(dict):
	for key in dict.keys():
		dict[key] = ' '.join(dict[key].split())
		# eliminate \n
		dict[key] = dict[key].replace('\n', ' ')
	return dict

naics_dict = eliminate_multimple_spaces_and_newlines(naics_dict)
# print(naics_dict)

import os
import json

# create directory for promts
def create_promts_dir():
	try:
		os.mkdir('promts')
	except FileExistsError:
		pass
 
def create_files(df=buisness_cat):
	for i in range(len(df)):
		name = df.iloc[i, 0]
		description = df.iloc[i, 1]
		promt = f"""Hi. I have a interesting challange for you today. Considering the following buisness taxonomy where i give the description of a buisness category and its name in parantheses: {naics_dict}. Match the following business category {name} with description {description} to the best 3 from the above taxonomy.Also give the confidency score for the 3 predicted matches.
		"""
		with open(f'promts/{i}_{name}.txt', 'w') as f:
			f.write(promt)

def create_dict2(df):
    return dict(zip(df['label'], df['description']))

buisness_cat_dict = create_dict2(buisness_cat)
buisness_cat_dict = eliminate_multimple_spaces_and_newlines(buisness_cat_dict)
# print(buisness_cat_dict)

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')
passage_embedding = model.encode(list(naics_dict.values()))


result_dict = {}


for i in range(len(buisness_cat_dict)):
	description = list(buisness_cat_dict.values())[i]
	query_embedding = model.encode(description)
	cosine_scores = util.pytorch_cos_sim(query_embedding, passage_embedding)
	top_three_pairs = [None, None, None]
	max_score_three = [0, 0, 0]
	# keep a list of the top three pair based on the max three scores
	for j in range(len(cosine_scores[0])):
		if cosine_scores[0][j] > min(max_score_three):
			index = max_score_three.index(min(max_score_three))
			max_score_three[index] = cosine_scores[0][j]
			top_three_pairs[index] = {'index': j, 'score': cosine_scores[0][j]}
   
	# the key is the name of the buisness category and the value is a list of the top three nasics category and their scores
	result_dict[list(buisness_cat_dict.keys())[i]] = [(list(naics_dict.keys())[pair['index']], pair['score'].item()) for pair in top_three_pairs]
	# print(str(list(buisness_cat_dict.keys())[i]) + ": " + str(result_dict[list(buisness_cat_dict.keys())[i]]))

# print(result_dict)
# write to json
# with open('result.json', 'w') as f:
# 	json.dump(result_dict, f)

# for each buisness category, take its description and run similarity with all the top three naics categories in the result_dict
# and create a result file with top match and score
final_result = {}
for key, value in result_dict.items():
	buisness_cat_description = buisness_cat_dict[key]
	top_3 = [value[i][0] for i in range(3)]
	query_embedding = model.encode(buisness_cat_description)
	passage_embedding = model.encode(top_3)
	cosine_scores = util.pytorch_cos_sim(query_embedding, passage_embedding)
	max_score = 0
	max_index = 0
	for i in range(len(cosine_scores[0])):
		if cosine_scores[0][i] > max_score:
			max_score = cosine_scores[0][i]
			max_index = i
	final_result[key] = (top_3[max_index], max_score.item())
	print(key + ": " + str(final_result[key]))
 
# write to json
with open('final_result.json', 'w') as f:
	json.dump(final_result, f)
	
