import json
from utils import tokenize, bag_of_words, stem
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import NeuralNet

class ChatDataset(Dataset):
  def __init__(self):
    self.n_samples = len(X_train)
    self.x_data = X_train
    self.y_data = y_train

  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.n_samples




with open('intents.json', 'r') as file:
  intents = json.load(file)

all_words = []
tags = []                             # all the categories
xy = []                               # zip processed arrs with tag
for intent in intents['intents']:
  tag = intent['tag']
  tags.append(tag)
  for pattern in intent['patterns']:
    processed = tokenize(pattern)
    all_words.extend(processed)
    xy.append((processed, tag))

ignored = [',', '.', '?', '!']
all_words = [stem(word) for word in all_words if word not in ignored]   # stem all the words
all_words = sorted(set(all_words))    # get rid of duplicates
tags = sorted(set(tags))              # get rid of duplicates

X_train = []
y_train = []
for (processed_sentence, tag) in xy:
  bag = bag_of_words(processed_sentence, all_words)
  X_train.append(bag)

  label = tags.index(tag)
  y_train.append(label)


# parameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 800


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')


data = {
  "model_state": model.state_dict(),
  "input_size": input_size,
  "output_size": output_size,
  "hidden_size": hidden_size,
  "all_words": all_words,
  "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'training done, save to {FILE}')