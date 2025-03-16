import pandas as pd
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import spacy

# Initialize SpaCy for NER
nlp = spacy.load("en_core_web_sm")

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Preprocessing
    data['response_tweet_id'] = data['response_tweet_id'].astype(str)
    data['tweet_id'] = data['tweet_id'].astype(str)
    customer_queries = data[data['inbound'] == True]
    company_responses = data[data['inbound'] == False]
    merged_data = pd.merge(
        customer_queries,
        company_responses[['tweet_id', 'text']],
        left_on='response_tweet_id',
        right_on='tweet_id',
        how='left',
        suffixes=('_customer', '_company')
    )
    conversation_data = merged_data[['text_customer', 'text_company']].dropna()
    return conversation_data

# Load your dataset (adjust the path as needed)
dataset_name = "sample.csv"  # Update with your file path
conversation_data = load_data(dataset_name)

# Split into training and testing
X = conversation_data['text_customer']
y = conversation_data['text_company']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine y_train and y_test for consistent encoding
all_labels = pd.concat([y_train, y_test])

# Encode labels using the combined dataset
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Transform train and test labels
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert encoded labels to tensors
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

# Tokenize the text
def tokenize_data(text_data):
    return tokenizer(text_data, padding=True, truncation=True, return_tensors="pt", max_length=512)

# Prepare the dataset for BERT
train_encodings = tokenize_data(X_train.tolist())
test_encodings = tokenize_data(X_test.tolist())

# Prepare the dataset for training BERT
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, y_train_tensor)
test_dataset = Dataset(test_encodings, y_test_tensor)

# Train BERT
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # Adjust epochs as necessary
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate BERT
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        
        predictions.extend(predicted_labels.numpy())
        true_labels.extend(labels.numpy())

from sklearn.metrics import classification_report
print(classification_report(true_labels, predictions))

# Intent Recognition using Hugging Face
intent_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Flask API
app = Flask(__name__)
@app.route('/chat', methods=['POST'])
def chatbot():
    user_query = request.json.get("query")  # Expect JSON input with the query field
    
    # Intent recognition
    intent_result = intent_classifier(user_query)
    predicted_intent = intent_result[0]['label']
    
    # Named Entity Recognition (NER)
    doc = nlp(user_query)
    entities = {ent.text: ent.label_ for ent in doc.ents}
    
    # Find response using BERT model
    inputs = tokenizer(user_query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    response = label_encoder.inverse_transform([predicted_class])[0]
    
    return jsonify({
        "query": user_query,
        "intent": predicted_intent,
        "entities": entities,
        "response": response
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
