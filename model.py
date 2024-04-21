import torch
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    BertTokenizerFast,
    BertForSequenceClassification,
    BertTokenizer,
    AutoTokenizer
)
from huggingface_hub import HfFolder, notebook_login
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_dataset=pd.read_csv("round5.csv")

model_id = "bert-base-uncased"

from sklearn.model_selection import train_test_split

# Split the dataset into train and test sets
train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_dataset)
test_dataset = Dataset.from_pandas(test_dataset)
dataset = Dataset.from_pandas(full_dataset)


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
unique_classes = full_dataset['predicted'].unique()


def tokenize_function(examples):
    return tokenizer(examples['concatenated_text'], padding="max_length", truncation=True)

train_dataset = train_dataset.rename_column('predicted', 'labels')
test_dataset = test_dataset.rename_column('predicted', 'labels')
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

model = BertForSequenceClassification.from_pretrained(model_id, num_labels=len(unique_classes))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

from sklearn.preprocessing import OneHotEncoder
import torch

def tokenize_function(examples):
    return tokenizer(examples['concatenated_text'], padding="max_length", truncation=True)

# Rename 'predicted' column to 'labels'
train_dataset = train_dataset.rename_column('predicted', 'labels')
test_dataset = test_dataset.rename_column('predicted', 'labels')

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# One-hot encode the 'labels' column
encoder = OneHotEncoder(sparse=False)
train_labels = encoder.fit_transform(train_dataset['labels'].to_numpy().reshape(-1, 1))
test_labels = encoder.transform(test_dataset['labels'].to_numpy().reshape(-1, 1))

# Replace 'labels' column in the datasets with one-hot encoded labels
train_dataset = train_dataset.remove_columns(['labels'])
train_dataset = train_dataset.add_column('labels', train_labels.tolist())
test_dataset = test_dataset.remove_columns(['labels'])
test_dataset = test_dataset.add_column('labels', test_labels.tolist())

# Set the format of the datasets
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize the model
model = BertForSequenceClassification.from_pretrained(model_id, num_labels=len(unique_classes))

# Define a custom loss function
def custom_loss(data, targets):
    targets = torch.tensor(targets, dtype=torch.float32)
    loss = torch.nn.BCEWithLogitsLoss()(data, targets)
    return loss

# Initialize the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_loss=custom_loss  # Use the custom loss function
)

# Train the model
trainer.train()


