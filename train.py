import json
import torch
from nltk_utils import tokenize,stem,bag_of_word
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open("intents.json","r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    if intent["tag"]== "default":
        import pdb;
        pdb.set_trace()
    if "patterns" in intent.keys():
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w,tag))

ignore_words = ["?","!",".",","]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []

for (pattern_sentence ,tag) in xy:
    bag = bag_of_word(pattern_sentence,all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epoch = 1000

# import pdb;pdb.set_trace()
dataset = ChatDataset()
print(input_size,len(all_words))
print(output_size,tags)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device is {device}")
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epoch):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch +1}/{num_epoch}, loss= {loss.item():4f}')
print(f'final loss= {loss.item():4f}')
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE  = "data.pth"
torch.save(data,FILE)

print(f"training complete. File saved to {FILE}")





