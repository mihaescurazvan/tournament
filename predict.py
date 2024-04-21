from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import OneHotEncoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import time
import requests

base_url = 'http://116.202.111.229:8000'
api_key = 'api-key'

headers = {
    'x-api-key': ''
}

def create_dict(df):
    return dict(zip(df['naics_label'], df['description']))


def eliminate_multimple_spaces_and_newlines(dict):
	for key in dict.keys():
		dict[key] = ' '.join(dict[key].split())
		# eliminate \n
		dict[key] = dict[key].replace('\n', ' ')
	return dict


def prepare_game():
# Initialize tokenizer and model
	model_id = "bert-base-uncased"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model5 = AutoModelForSequenceClassification.from_pretrained("checkpoint-6500")
	model4 = AutoModelForSequenceClassification.from_pretrained("round4-checkpoint-6500")
	naics = pd.read_excel('Naics3 (label) taxonomy.xlsx')
	naics_dict = create_dict(naics)
	naics_dict = eliminate_multimple_spaces_and_newlines(naics_dict)
	# Load dataset and prepare labels
	full_dataset = pd.read_csv("round5.csv")
	train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
	labels = np.array(train_dataset['predicted']).reshape(-1, 1)  # Ensure labels are correctly shaped

	# Fit the OneHotEncoder once
	encoder = OneHotEncoder(sparse=False)
	encoder.fit(labels)
 
	model5.eval()
	model4.eval()
	model = SentenceTransformer('all-mpnet-base-v2')
	return model5, model4, tokenizer, encoder, naics_dict, model

 

model5, model4, tokenizer, encoder, naics_dict, model = prepare_game()
print("Ready to start the game")
abstain = False
while True:
    # if read from stdin break
    if input() == 'break':
        break

# hint = ""
# for i in range(5):
# 	response = requests.get(f"{base_url}/evaluate/hint", headers=headers)
# 	print(i, response)
# 	if hint == "":
# 		hint = response.json()['hint']
# 	else:
# 		hint = hint + ";" + response.json()['hint']
# 	# Tokenize the input text
# 	inputs = tokenizer(hint, return_tensors="pt", padding=True, truncation=True, max_length=512)
# 	# Move inputs to the same device as the model
# 	inputs = {key: value for key, value in inputs.items()}
# 	# Perform inference
# 	with torch.no_grad():
# 		outputs5 = model5(**inputs)
# 		logits5 = outputs5.logits
		
# 		outputs4 = model4(**inputs)
# 		logits4 = outputs4.logits
# 	# Softmax to get probabilities
# 	probabilities5 = torch.nn.functional.softmax(logits5, dim=-1).cpu().numpy()
# 	probabilities4 = torch.nn.functional.softmax(logits4, dim=-1).cpu().numpy()
# 	predicted_index5 = np.argmax(probabilities5, axis=1)
# 	predicted_one_hot5 = np.zeros(probabilities5.shape)
# 	predicted_one_hot5[np.arange(len(probabilities5)), predicted_index5] = 1
# 	predicted_labels5 = encoder.inverse_transform(predicted_one_hot5)
# 	predicted_labels5 = predicted_labels5[0][0]
	
# 	predicted_index4 = np.argmax(probabilities4, axis=1)
# 	predicted_one_hot4 = np.zeros(probabilities4.shape)
# 	predicted_one_hot4[np.arange(len(probabilities4)), predicted_index4] = 1
# 	predicted_labels4 = encoder.inverse_transform(predicted_one_hot4)
# 	predicted_labels4 = predicted_labels4[0][0]
	
# 	# make cosine similarity between the hint and description of the predicted_labels and choose the one with the highest similarity
# 	query_embedding = model.encode(hint)
# 	passage_embedding = model.encode(naics_dict[predicted_labels5])
# 	cosine_scores = util.pytorch_cos_sim(query_embedding, passage_embedding)
	
# 	passage_embedding2 = model.encode(naics_dict[predicted_labels4])
# 	cosine_scores2 = util.pytorch_cos_sim(query_embedding, passage_embedding2)
	
# 	if cosine_scores > cosine_scores2:
# 		predicted_labels = predicted_labels5
# 	else:
# 		predicted_labels = predicted_labels4
# 		cosine_scores = cosine_scores2
	
# 	if i + 1 >= 3 and cosine_scores < 0.15 and abstain == False:
# 		abstain = True
# 		data = {
# 			'answer': 'abstain'
# 		}
# 		response = requests.post(f"{base_url}/evaluate/answer", json=data, headers=headers)
# 		continue
# 	elif i + 1 >= 2 and cosine_scores < 0.1 and abstain == False:
# 		abstain = True
# 		data = {
# 			'answer': 'abstain'
# 		}
# 		response = requests.post(f"{base_url}/evaluate/answer", json=data, headers=headers)
# 		continue
# 	data = {
# 		'answer': predicted_labels
# 	}
# 	response = requests.post(f"{base_url}/evaluate/answer", json=data, headers=headers)
# response = requests.get(f"{base_url}/evaluate/hint", headers=headers)
hint = 'Lipinski Productions is a Pittsburgh-based company dedicated to providing the highest level of media production services in video editing, media conversion, graphic design, web design, and more....'
# Tokenize the input text
inputs = tokenizer(hint, return_tensors="pt", padding=True, truncation=True, max_length=512)
# Move inputs to the same device as the model
inputs = {key: value for key, value in inputs.items()}
# Perform inference
with torch.no_grad():
	outputs5 = model5(**inputs)
	logits5 = outputs5.logits
	
	outputs4 = model4(**inputs)
	logits4 = outputs4.logits
# Softmax to get probabilities
probabilities5 = torch.nn.functional.softmax(logits5, dim=-1).cpu().numpy()
probabilities4 = torch.nn.functional.softmax(logits4, dim=-1).cpu().numpy()
predicted_index5 = np.argmax(probabilities5, axis=1)
predicted_one_hot5 = np.zeros(probabilities5.shape)
predicted_one_hot5[np.arange(len(probabilities5)), predicted_index5] = 1
predicted_labels5 = encoder.inverse_transform(predicted_one_hot5)
predicted_labels5 = predicted_labels5[0][0]

predicted_index4 = np.argmax(probabilities4, axis=1)
predicted_one_hot4 = np.zeros(probabilities4.shape)
predicted_one_hot4[np.arange(len(probabilities4)), predicted_index4] = 1
predicted_labels4 = encoder.inverse_transform(predicted_one_hot4)
predicted_labels4 = predicted_labels4[0][0]

# make cosine similarity between the hint and description of the predicted_labels and choose the one with the highest similarity
query_embedding = model.encode(hint)
passage_embedding = model.encode(naics_dict[predicted_labels5])
cosine_scores = util.pytorch_cos_sim(query_embedding, passage_embedding)

passage_embedding2 = model.encode(naics_dict[predicted_labels4])
cosine_scores2 = util.pytorch_cos_sim(query_embedding, passage_embedding2)

if cosine_scores > cosine_scores2:
	predicted_labels = predicted_labels5
else:
	predicted_labels = predicted_labels4
	cosine_scores = cosine_scores2  
	
        


# # Tokenize the input text
# inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)



# # Move inputs to the same device as the model
# inputs = {key: value for key, value in inputs.items()}

# # Perform inference
# with torch.no_grad():
#     outputs5 = model5(**inputs)
#     logits5 = outputs5.logits
    
#     outputs4 = model4(**inputs)
#     logits4 = outputs4.logits
    
#     # outputs3 = model3(**inputs)
#     # logits3 = outputs3.logits

# # Softmax to get probabilities
# probabilities5 = torch.nn.functional.softmax(logits5, dim=-1).cpu().numpy()
# probabilities4 = torch.nn.functional.softmax(logits4, dim=-1).cpu().numpy()
# # probabilities3 = torch.nn.functional.softmax(logits3, dim=-1).cpu().numpy()

# predicted_index5 = np.argmax(probabilities5, axis=1)
# predicted_one_hot5 = np.zeros(probabilities5.shape)
# predicted_one_hot5[np.arange(len(probabilities5)), predicted_index5] = 1
# predicted_labels5 = encoder.inverse_transform(predicted_one_hot5)
# predicted_labels5 = predicted_labels5[0][0]

# predicted_index4 = np.argmax(probabilities4, axis=1)
# predicted_one_hot4 = np.zeros(probabilities4.shape)
# predicted_one_hot4[np.arange(len(probabilities4)), predicted_index4] = 1
# predicted_labels4 = encoder.inverse_transform(predicted_one_hot4)
# predicted_labels4 = predicted_labels4[0][0]

# # predicted_index3 = np.argmax(probabilities3, axis=1)
# # predicted_one_hot3 = np.zeros(probabilities3.shape)
# # predicted_one_hot3[np.arange(len(probabilities3)), predicted_index3] = 1
# # predicted_labels3 = encoder.inverse_transform(predicted_one_hot3)
# # predicted_labels3 = predicted_labels3[0][0]

# # choose the one who appears the most, if there is a tie, choose the predicted_labels5
# predicted_labels = predicted_labels5
# if predicted_labels4 == predicted_labels:
# 	predicted_labels = predicted_labels4
# # elif predicted_labels3 == predicted_labels:
# # 	predicted_labels = predicted_labels3
 
# print(predicted_labels)

# # curl -H 'x-api-key: lnzYdcdRr8RiBniVUAXPZlidQAnhDggd' -H 'Content-Type: application/json' -X POST -d '{"answer": "Religious, Grantmaking, Civic, Professional, and Similar Organizations"}' http://116.202.111.229:8000/evaluate/answer
# # curl -H 'x-api-key: lnzYdcdRr8RiBniVUAXPZlidQAnhDggd' http://116.202.111.229:8000/evaluate/hint
# # curl -H 'x-api-key: lnzYdcdRr8RiBniVUAXPZlidQAnhDggd' http://116.202.111.229:8000/evaluate/reset
# # curl -H 'x-api-key: lnzYdcdRr8RiBniVUAXPZlidQAnhDggd' -H 'Content-Type: application/json' -X POST -d '{"answer": "abstain"}' http://116.202.111.229:8000/evaluate/answer