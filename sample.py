#导入库
import os, torch
import tokenizer
from model import nanoModel
#设备与权重地址
device = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE = os.path.dirname(os.path.abspath(__file__))
#加载权重
ckpt = torch.load(os.path.join(HERE, 'ckpt.pt'), map_location=device, weights_only=False)

m = nanoModel(**ckpt['config']).to(device)
m.load_state_dict(ckpt['model'])
print(f"loaded: step={ckpt['step']}  val={ckpt['val']:.4f}")
#输出
#提示词
prompt = "SCENE I"
ids = [tokenizer.tokenizerEncoding(prompt)] 
idx = torch.tensor(ids, dtype=torch.long, device=device)
out = m.generate(idx, max_new_tokens=1000)
print(tokenizer.tokenizerDecoding(out[0].tolist()))