from sentence_transformers import SentenceTransformer, util
import os
import json
import pandas as pd
import numpy as np

# read the final_result.json
def read_final_result():
	with open('result.json', 'r') as f:
		return json.load(f)

# read the final_result.json
dict_mathces = read_final_result()
df = pd.read_csv("cutted_tournament_dataset.csv")
# df.set_index('commercial_name', inplace=True)
print(df.head())

model = SentenceTransformer('all-mpnet-base-v2')
naics = pd.read_excel('Naics3 (label) taxonomy.xlsx')

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

# for company in df.index:
# 	print(company)
# 	description = df.at[company, 'description']
# 	print(description)
# 	query_embedding = model.encode(description)
# 	key = df.at[company]['main_business_category']
# 	print(key)
# 	value = dict_mathces[key]
# 	top_3 = [value[i][0] for i in range(3)]
# 	top_3_description = [naics_dict[i] for i in top_3]
# 	for i in range(3):
# 		print(f"Predicted: {top_3[i]} with description: {top_3_description[i]}")
	
targeted_df = pd.DataFrame(columns=['company', 'description', 'main_business_category', 'predicted', 'confidence_score'])

for i in range(len(df)):
	name = df.iloc[i, 0]
	description = df.iloc[i, 1]
	query_embedding = model.encode(description)
	key = df.iloc[i, 2]
	value = dict_mathces[key]
	top_3 = [value[i][0] for i in range(3)]
	top_3_description = [naics_dict[i] for i in top_3]
	passage_embedding = model.encode(top_3_description)
	print(passage_embedding.shape)
	cosine_scores = util.pytorch_cos_sim(query_embedding, passage_embedding)
	print(cosine_scores)
	predicted = top_3[np.argmax(cosine_scores)]
	#  find max confidence score
	confidence_score = cosine_scores[0, np.argmax(cosine_scores)]
	#  use concat
	
	targeted_df = pd.concat([targeted_df, pd.DataFrame([[name, description, key, predicted, confidence_score]], columns=['company', 'description', 'main_business_category', 'predicted', 'confidence_score'])])
	# print(f"Predicted: {predicted} with description: {naics_dict[predicted]} for company {name} with description {description} with confidence score {confidence_score}")
	if i % 1000 == 0:
		targeted_df.to_csv('targeted_dataset.csv', index=False)
 
 
targeted_df.to_csv('targeted_dataset.csv', index=False)
	