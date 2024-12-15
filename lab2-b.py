import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model


# 1. Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load dataset
dataset = load_dataset("imdb")


# Sample 10% from train and split test into 80% for test, 20% for validation
train_data = list(zip(dataset['train']['text'], dataset['train']['label']))
train_data_sampled = np.random.choice(len(train_data), size=int(len(train_data) * 0.1), replace=False)
train_data_sampled = [train_data[i] for i in train_data_sampled]


test_data = list(zip(dataset['test']['text'], dataset['test']['label']))
val_size = int(len(test_data) * 0.2)
test_data, val_data = random_split(test_data, [len(test_data) - val_size, val_size])


# Unzip data into features and labels
X_train, y_train = zip(*train_data_sampled)
X_test, y_test = zip(*test_data)
X_val, y_val = zip(*val_data)


# Print lengths
print(len(X_train), len(X_test), len(X_val))


# Tokenization and Data Preparation
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)  # Remove batch dimension
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token


# Create DataLoaders
train_dataset = SentimentDataset(X_train, y_train, tokenizer)
val_dataset = SentimentDataset(X_val, y_val, tokenizer)
test_dataset = SentimentDataset(X_test, y_test, tokenizer)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# Define GPT-2-based model with PEFT LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1, task_type="SEQ_CLS"
)


base_model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
# Update the pad_token_id in the model config
base_model.config.pad_token_id = base_model.config.eos_token_id
model = get_peft_model(base_model, lora_config)
model = model.to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 3
train_losses = []
val_losses = []


for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}")


    # Validation
    model.eval()
    val_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, axis=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())


    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")


# Plot training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()






# Evaluate the model on the test set
predictions = []
true_labels = []


model.eval()
with torch.no_grad():
    for i, (input_ids, attention_mask, labels) in enumerate(test_loader):
        if i >= 10:
          break
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, axis=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")


# Print the classification report
print("\nClassification Report:\n")
print(classification_report(true_labels, predictions))
