#导入库
import os, torch
import torch.nn.functional as F
import tokenizer
from model import nanoModel

#编码训练集
HERE = os.path.dirname(os.path.abspath(__file__))

#处理数据集
texts = open(os.path.join(HERE, 'shakespeare_complete_pg100.txt'), encoding='utf-8').read()

cache = os.path.join(HERE, 'tokens_3000.pt')
if os.path.exists(cache):
    data = torch.load(cache)
    print('loaded')
else:
    data = torch.tensor(tokenizer.tokenizerEncoding(texts), dtype=torch.long)
    torch.save(data, cache)
    print('tokenized')
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
#超参数
#模型大小控制
block_size = 256
batch_size = 64
vocab_size = tokenizer.LENTH
n_embd = 384
hidden_size = n_embd*4
num_heads = 8
n_layer = 6 
#训练控制
dropout = 0.3
max_steps, aver_interval = 5001, 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


#切数据
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device) 
    return x,y
#模型存储
config = dict(vocab_size=tokenizer.LENTH, n_embd=n_embd, n_layer=n_layer,
              num_heads=num_heads, block_size=block_size, dropout=dropout,
              hidden_size = hidden_size)
model = nanoModel(**config).to(device)
#优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#最优值
best_val = float('inf')
#可视化训练
@ torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(100)
        for k in range(100):
            X,Y = get_batch(split)
            logit , loss = model(X,Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out 

#训练主体
for step in range(max_steps):
    if step >= 0.5*max_steps:
        lr = 3e-5
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if step % aver_interval == 0:
        losses = estimate_loss()
        print(f'step{step}: train {losses["train"]:.4f} | val {losses["val"]:.4f}')
        v = losses['val'].item()
        #存最优权重
        if v < best_val:
            best_val = v
            torch.save({'model': model.state_dict(), 'config': config,
                        'step': step, 'val': v}, os.path.join(HERE, 'ckpt.pt'))
        
#状态检测
print('训练完成,best val =', round(best_val, 4))