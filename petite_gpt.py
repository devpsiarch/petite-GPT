# -*- coding: utf-8 -*-
"""petite_gpt.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10w0d1DGL_rGMlYEwAwtT_B8J9HutbFFP
"""

#gets the dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt



#reads the whole text from the dataset
with open("input.txt",'r',encoding='utf-8') as file:
  text = file.read()

print("length of the dataset in char is ",len(text))
print(text[:1000])

#gets the outputs that are expected
chars = sorted(list(set(text)))
vocabolary_size = len(chars)
print(''.join(chars))
print(vocabolary_size)

#maps the chars to numbers AKA tokinze them
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hello friends"))
print(decode(encode("hello this time am ready")))

#encode the all the dataset to a tensore
import torch
data = torch.tensor(encode(text),dtype=torch.long)

print(data.shape,data.dtype)
print(data[:1000])

#spliting the dataset into training data and a test data (validation)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

#defining block size aka context lenght
block_size = 8
print(train_data[:block_size+1])

#more explinication of the context lenght
x = train_data[:block_size]
y = train_data[1:block_size+1]
for i in range(block_size):
  context = x[:i+1]
  expected = y[i]
  print("when context is {context} expected is {expected}",context,expected)

#the batch demension
batch_size = 4 # how many proccess to be run at the same time
torch.manual_seed(1337)
eval_iter = 5000

def get_batch(split):
  data = train_data if split == "train" else test_data
  ix = torch.randint(len(data)-block_size,(batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y

@torch.no_grad()
def estimate_loss():
  out = {}
  m.eval()
  for split in ["train","test"]:
    losses = torch.zeros(eval_iter)
    for k in range(eval_iter):
      x,y = get_batch(split)
      logits , loss = m.feedforward(x,y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  m.train()
  return out




xb,yb = get_batch("train")
print("input")
print(xb.shape)
print(xb)
print("output")
print(yb.shape)
print(yb)


for b in range(batch_size):
  for t in range(block_size):
    context = xb[b,:t+1]
    expected = yb[b,t]
    print(f"when the context is {context.tolist()} the expected is {expected}")

import torch.nn as nn
from torch.nn import functional as F

class BigrameLM(nn.Module):
  def __init__(self,vocabolary_size):
    super().__init__()
    self.token_embadding_table = nn.Embedding(vocabolary_size,vocabolary_size)

  def feedforward(self,idx,expected=None):
    #basiclly what we got from feeding forward into the model
    logits = self.token_embadding_table(idx)

    if expected == None:
      cost = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      expected = expected.view(B*T)

      cost = F.cross_entropy(logits,expected)

    return logits,cost

  def generate(self,idx,max_token):
    for _ in range(max_token):
      #feedforward the model
      logits , loss = self.feedforward(idx)
      logits = logits[:,-1,:] # makes it (B,C)
      #picks that char based on probabiliry
      probs = F.softmax(logits,dim=1)
      idx_next = torch.multinomial(probs,num_samples=1)
      idx = torch.cat((idx,idx_next),dim=1)
    return idx

m = BigrameLM(vocabolary_size)
logits,cost = m.feedforward(xb,yb)
print(logits.shape)
print(f"the loss/cost of the model is {cost}")


print(decode(m.generate(torch.zeros((1,1),dtype=torch.long),max_token=100)[0].tolist()))

#training the model
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)
batch_size = 32
for steps in range(eval_iter):
  xb,yb = get_batch("train")
  logits , loss = m.feedforward(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


print(estimate_loss())
#test after the training
print(decode(m.generate(torch.zeros((1,1),dtype=torch.long),max_token=500)[0].tolist()))

"""# Math trick for self-attention"""

