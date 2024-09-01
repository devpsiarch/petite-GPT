import torch
import torch.nn as nn
from torch.nn import functional as F

#parameters
batch_size = 64 
block_size = 256
max_iter = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iter = 200
n_embed = 384
n_layers = 6
n_heads = 6
dropout = 0.2
#getting the dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt",'r',encoding='utf-8') as file:
  text = file.read()



#gets the outputs that are expected
chars = sorted(list(set(text)))
vocabolary_size = len(chars)

#print(''.join(chars))
#print(vocabolary_size)


#maps the chars to numbers AKA tokinze them
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#test our very simple encode and decoder
#print(encode("hello friends"))
#print(decode(encode("hello this time am ready")))

data = torch.tensor(encode(text),dtype=torch.long)

#printf a sample from the dataset
#print(data.shape,data.dtype)
#print(data[:1000])

#spliting the dataset into training data and a test data (validation)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
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

class Head(nn.Module):
    """single self-attention head"""
   
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias= False)
        self.query = nn.Linear(n_embed,head_size,bias= False)
        self.value = nn.Linear(n_embed,head_size,bias= False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v 
        return out

class Multi_attention_head(nn.Module):
    def __init__(self,n_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self,n_embed,n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = Multi_attention_head(n_heads,head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigrameLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabolary_size,n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed,n_heads=n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed,vocabolary_size)



    def feedforward(self,idx,expected=None):
        B, T = idx.shape
        #basiclly what we got from feeding forward into the model
        token_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lm_head(x)

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
            idx_con = idx[:,-block_size:]
            logits , loss = self.feedforward(idx_con)
            logits = logits[:,-1,:] # makes it (B,C)
            #picks that char based on probabiliry
            probs = F.softmax(logits,dim=1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx



model = BigrameLM()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

#training the model
for steps in range(max_iter):
    xb,yb = get_batch("train")
    logits , loss = m.feedforward(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print(estimate_loss())
#test after the training
print(decode(m.generate(torch.zeros((1,1),dtype=torch.long),max_token=500)[0].tolist()))
