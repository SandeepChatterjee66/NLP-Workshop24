
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection
import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import gensim.downloader as api


# 1. Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 2. Download the Word2Vec model from gensim's API (small model)
def load_word2vec_model():
    # Load the pre-trained Word2Vec model (Google's 100-dimensional vectors)
    model = api.load("word2vec-google-news-300")
    print("Word2Vec model loaded successfully!")
    return model


# Load Word2Vec model
word2vec_model = load_word2vec_model()

# Convert text to Word2Vec embeddings
def text_to_embedding(text, model):
    embeddings = [model[word] for word in text.split() if word in model]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)


OR


import nltk
import re
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')


def text_to_embedding(text, model):
    # Replace non-alphabetic characters with spaces
    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())  # Keep only alphabets and spaces
    tokens = word_tokenize(cleaned_text)  # Tokenize the cleaned text
    embeddings = [model[word] for word in tokens if word in model]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)




# Prepare features (text -> word embeddings) and labels (positive/negative sentiment)
X = np.array([text_to_embedding(text, word2vec_model) for text in dataset['text']])
y = np.array(dataset['label'])


# Fix issue with converting labels to integers
y = y.astype(int)


# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)


# Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
val_data = TensorDataset(X_val_tensor, y_val_tensor)


train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)


# 4. Define MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Binary classification (positive/negative)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


# Instantiate model and move it to the GPU
model = MLPModel(input_dim=X_train.shape[1]).to(device)
print(model)  # This will print out the architecture of the neural network
from torchsummary import summary
summary(model, (X_train.shape[1],))  # Show model architecture summary




# 5. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 6. Train the model and plot the loss
train_losses = []
val_losses = []
for epoch in range(10):  # Number of epochs
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
   
    train_losses.append(running_loss / len(train_loader))


    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
   
    val_losses.append(val_loss / len(val_loader))


    print(f"Epoch [{epoch+1}/10], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")


# 7. Plot the training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


# 8. Save and load the model
torch.save(model.state_dict(), 'sentiment_model.pth')  # Save model
print("Model saved!")


# Load the model
loaded_model = MLPModel(input_dim=X_train.shape[1]).to(device)
loaded_model.load_state_dict(torch.load('sentiment_model.pth'))
loaded_model.eval()
# 9. Test on sample inputs
sample_reviews = [
    "They looked quite wonderful, it was thrilling",
    "It was a waste of time. Totally boring and predictable."
]


# Convert the sample reviews to embeddings
sample_reviews_embeddings = np.array([text_to_embedding(review, word2vec_model) for review in sample_reviews])
sample_reviews_tensor = torch.tensor(sample_reviews_embeddings, dtype=torch.float32).to(device)


with torch.no_grad():
    outputs = loaded_model(sample_reviews_tensor)
    predicted_labels = torch.argmax(outputs, dim=1)
    for review, label in zip(sample_reviews, predicted_labels):
        sentiment = "Positive" if label == 1 else "Negative"
        print(f"Review: {review}\nSentiment: {sentiment}\n")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# 9. Evaluate the model on the test dataset
# Assuming you have a test dataset
# If not, you can split the data as test dataset or use part of the validation dataset


# Example: Using the validation set as a "test" dataset (just for demonstration)
test_loader = DataLoader(val_data, batch_size=64, shuffle=False)


# Predictions and True Labels
predictions = []
true_labels = []


model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
        predictions.extend(predicted.cpu().numpy())  # Move to CPU for evaluation
        true_labels.extend(labels.cpu().numpy())  # Move to CPU for evaluation


# Compute Evaluation Metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='binary')
recall = recall_score(true_labels, predictions, average='binary')
f1 = f1_score(true_labels, predictions, average='binary')


# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, predictions))