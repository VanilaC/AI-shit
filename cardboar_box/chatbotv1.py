import numpy as np
import re
import json
import random
import os

C_PATH = os.path.dirname(__file__)
F_PATHone = os.path.join(C_PATH, 'embeddings.json')
F_PATHtwo = os.path.join(C_PATH, 'data_training.json')

with open(F_PATHone, 'r') as f:
    embeddings = json.load(f)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def get_e(word):
    embedding = embeddings.get(word, None)
    return embedding

def get_emb(sentence):
    sentence = sentence.lower()
    words = sentence.split()
    encoded_sentence = []

    for word in words:
        encoded_sentence.append(get_e(word))
    encoded_sentence = np.array(encoded_sentence)

    flat_one_hot = encoded_sentence.reshape(-1)
    flat_one_hot = np.append(flat_one_hot,[0] *(50-len(flat_one_hot)))

    return flat_one_hot

with open(F_PATHtwo) as file:
    data = json.load(file)
patt = []
res = []
for j in data["intents"]:
    patt.append(j["patterns"])
    res.append(j["responses"])




num_hidden_neurons = 12
lr = 0.5

i_to_hw = np.random.randn(50, num_hidden_neurons)
i_to_hb = np.zeros((1, num_hidden_neurons))

h_to_ow = np.random.randn(num_hidden_neurons, 10)
h_to_ob = np.zeros((1, 10))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation(inputs, i_to_hw, i_to_hb, h_to_ow, h_to_ob):
    hidden = sigmoid(np.dot(inputs, i_to_hw) + i_to_hb)
    output = sigmoid(np.dot(hidden, h_to_ow) + h_to_ob)
    return output, hidden

def backward_propagation(inputs, targets, predict, hidden, i_to_hw, h_to_ow,h_to_ob,i_to_hb):
    delta_output = (targets - predict) * predict * (1 - predict)
    delta_hidden = np.dot(delta_output, h_to_ow.T) * hidden * (1 - hidden)

    h_to_ow += lr * np.dot(hidden.T, delta_output)
    h_to_ob += lr * np.sum(delta_output, axis=0, keepdims=True)
    i_to_hw += lr * np.dot(inputs.reshape(50, 1), delta_hidden)
    i_to_hb += lr * np.sum(delta_hidden, axis=0, keepdims=True)

    return i_to_hw, i_to_hb, h_to_ow, h_to_ob
def train(inputs,targets):
    global i_to_hw
    global i_to_hb
    global h_to_ow
    global h_to_ob
    predict, hidden = forward_propagation(inputs, i_to_hw, i_to_hb, h_to_ow, h_to_ob,)
    error = np.mean(np.square(predict - targets))

    i_to_hw, i_to_hb, h_to_ow, h_to_ob = backward_propagation(inputs, targets, predict, hidden, i_to_hw, h_to_ow,h_to_ob,i_to_hb)
    return error

def get_one_hot(num):
    one_hot = [0] * 10
    one_hot[num] = 1
    return one_hot
for p in range(1000):
    b = 0
    for j in patt:
        for n in j:
            ans = get_one_hot(b)
            error = train(get_emb(n),ans)
        b += 1


print("ready!","error rate:",error)
running = True
while running:
    i = input('You:')
    if i == 'x':
        break
    inputs = get_emb(i)
    predict, hidden = forward_propagation(inputs, i_to_hw, i_to_hb, h_to_ow, h_to_ob)
    answer = np.round(predict)
    print(softmax(predict[0]))
    try:
        print("Neurul:",random.choice(res[np.where(answer[0] == 1)[0][0]]))
    except:
        print("sorry but i dont know what are you talking about?")
        print("activation:",answer)