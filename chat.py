import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_word, tokenize
from nltk import pos_tag
from context import extract_order_items,validate_ordered_items
from db_utils import insert_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('intents.json','r') as f:
    intents = json.load(f)

FILE  = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Lets Chat! (type 'quit' to exit)")
context = {}
while True:
    sentence = input("You: ")
    if sentence=='quit':
        break
    sentence = tokenize(sentence)
    x = bag_of_word(sentence,all_words)
    x = x.reshape(1,x.shape[0])
    x = torch.from_numpy(x).to(device)

    output = model(x)
    _,predicted = torch.max(output,dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        print(prob.item())
        for intent in intents['intents']:
            if tag==intent["tag"]:
                print(tag)
                tagged_tokens = pos_tag(sentence)
                print(f"tagged_tokens : {tagged_tokens}")
                order_items = extract_order_items(tagged_tokens)
                print(f"order_items : {order_items}")
                context = validate_ordered_items(order_items,context,tag)
                print(f"context : {context}")
                if tag=="order complete":
                    order_id, final_cost =insert_data(context)
                    text = f"{final_cost} and your order id is {order_id}"
                    print(f"{bot_name}:{random.choice(intent['responses']) + text}")
                else:
                    print(f"{bot_name}:{random.choice(intent['responses'])}")

    else:
        resp = intents['intents'][5]["responses"]
        print("in else")
        print(f"{bot_name}: {resp[0]}")





