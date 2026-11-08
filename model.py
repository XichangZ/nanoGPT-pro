#导入库
import torch
import torch.nn as nn
from torch.nn import functional as F

#定义单头
class Head(nn.Module):
    def __init__(self, head_size,n_embd,block_size,dropout):
        super().__init__()
        self.heads_size = head_size
        self.key = nn.Linear(n_embd,head_size,bias = False)
        self.query = nn.Linear(n_embd,head_size,bias = False)
        self.value = nn.Linear(n_embd,head_size,bias = False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)),persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) / (self.heads_size**0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        attention = wei @ v
        return attention

#多头
class MultiHeadAttention(nn.Module):
    
    def __init__(self,num_heads,heads_size,n_embd,dropout,block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(heads_size,n_embd,block_size,dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out =  torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out 

#FFN
class FeedForward(nn.Module):
    def __init__(self, n_embd,hidden_size,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,n_embd),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

#完整Transformer解码器
class Block(nn.Module):
    def __init__(self, n_embd,num_heads,hidden_size,dropout,block_size):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads,head_size,n_embd,dropout,block_size)
        self.ffn = FeedForward(n_embd,hidden_size,dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self,x):
        x = self.sa(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x
#model
class nanoModel(nn.Module):
    def __init__(self,vocab_size,n_embd,block_size,num_heads,hidden_size,n_layer,dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) 
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        #主体
        self.blocks = nn.Sequential(*[Block(n_embd,num_heads,hidden_size,dropout,block_size) for _ in range(n_layer)]) 
        #初始化词义矩阵
        nn.init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding_table.weight, mean=0.0, std=0.02)
        #多层结束后的归一化
        self.ln_f = nn.LayerNorm(n_embd)
        self.lmhead = nn.Linear(n_embd,vocab_size)
        

    def forward(self,idx,targets = None):
        B,T = idx.shape
        #前置词义矩阵
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T,device = idx.device))
        x = tok_emb + pos_emb
        #模型层
        x = self.blocks(x)
        #输出层
        x = self.ln_f(x)
        logits = self.lmhead(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits,loss

    #生成模式
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  #加大数据量后发现长噪声变多，改进
            #防止超长词
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf') 
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        self.train()
        return idx
