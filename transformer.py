# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt

torch.manual_seed(42)

# %%
class SimpleTokenizer:
    def __init__(self, text_corpus):
        chars = sorted(list(set(text_corpus)))
        self.chars = ['<pad>', '<sos>', '<eos>'] + chars

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.vocab_size = len(self.chars)
        self.pad_token_id = self.stoi['<pad>']
        self.sos_token_id = self.stoi['<sos>']
        self.eos_token_id = self.stoi['<eos>']

    def encode(self, text):
        return [self.stoi[c] for c in text]
    
    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([self.itos[i] for i in indices if i < len(self.itos)])

# %%
class ToyDataset(Dataset):
    def __init__(self, text_data, tokenizer:SimpleTokenizer, max_len):
        self.data = text_data # 리스트 형태
        self.tokenizer = tokenizer # 토크나이저 위에
        self.max_len = max_len # 입력 고정 길이 (입력 통일)

    def __len__(self):
        return len(self.data) # 데이터에 몇개 있는지
    
    def __getitem__(self, idx):
        text = self.data[idx]

        encoded = self.tokenizer.encode(text)

        token_ids = [self.tokenizer.sos_token_id] + encoded + [self.tokenizer.eos_token_id]
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len] # 자르기

        pad_len = self.max_len - len(token_ids) # 남는 거
        token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len

        return torch.tensor(token_ids, dtype=torch.long)

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        '''
            d_model: 임베딩 차원
            max_len: 최대길이
        '''
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # [max_len, d_model] -> [1, max_len, d_model]
        self.register_buffer('pe', pe)
        self.pe: torch.Tensor
        # pe 텐서를 모델의 buffer로 등록, => state_dict에 포함되어 모델에 함께 관리
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


# %%
corpus = ["hello transformer", "jupyter notebook is good", "i love python"]
max_len = 12
d_model = 32
batch_size = 2

# Tokenizer 생성
tokenizer = SimpleTokenizer("".join(corpus))
print(f"vocab size: {tokenizer.vocab_size}")
print(f"dict: {tokenizer.stoi}")

dataset = ToyDataset(corpus, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size, shuffle= True)

batch_indices = next(iter(dataloader))
print(f"Input batch shape: {batch_indices.shape}")
print(f"첫 번째 문장: {tokenizer.decode(batch_indices[0])}")

# %%
embedding = nn.Embedding(tokenizer.vocab_size, d_model)
pos_encoder = PositionalEncoding(d_model, max_len)

embed_out = embedding(batch_indices)
final_out = pos_encoder(embed_out)
print(f"최종 Output shape: {final_out.shape}")
# print(embed_out)

# %%
pe_test = PositionalEncoding(d_model=128, max_len=100)
y = pe_test.pe.squeeze(0).numpy()

plt.figure(figsize=(10, 6))
plt.imshow(y, cmap='RdBu', aspect='auto')
plt.title("Positional Encoding Matrix")
plt.xlabel("Depth (d_model)")
plt.ylabel("Position (Time step)")
plt.colorbar()
plt.show()

# %%
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1) 
        # 마지막 축이 합 1되게 정규화
    
    def forward(self, q, k, v, mask=None):
        # q,k,v: [batch_size, n_heads, seq_len, head_dim]
        # qk^T : [batch_size, n_heads, seq_len, seq_len]

        scores = torch.matmul(q, k.transpose(-1, -2))
        # 마지막 두 차원만 곱함

        d_k = q.size(-1)
        scores = scores/math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        attn_weights = self.softmax(scores)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights


# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        # d_model: 모델 차원, n_heads: head 수
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 각 헤드 차원
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model가 n_heads의 배수 아님"

        # 파라미터화 행렬
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 선형 변환 및 헤드 분리
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, head_dim]
        # -> [batch, n_heads, seq_len, head_dim] (transpose)

        Q = self.w_q(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot product Attention
        # output shape: [batch, n_heads, seq_len, head_dim]
        output, attn_weights = self.attention(Q, K, V, mask=mask)
        output: torch.Tensor
        # Concat
        # 최종: [batch, seq_len, d_model]
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.w_o(output)
        return output, attn_weights




# %%
# 어텐션 테스트

d_model = 32
n_heads = 4
seq_len = 12
batch_size = 2

mha = MultiHeadAttention(d_model, n_heads)

# 가짜 position encoded embedded 데이터
x = torch.randn(batch_size, seq_len, d_model)

output, weights = mha(x, x, x, mask=None)

print(f"input shape: {x.shape}")
print(f"output shape: {output.shape}")

print (f"attn weights: {weights.shape}")
# [2, 4, 12, 12]

if x.shape == output.shape:
    print("차원 일치")
else:
    print("차원 불일치")

# %%
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # d_ff는 보통 d_model의 4배
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# %%
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 정규화 및 드롭아웃
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 잔차 연결 x + sublayer(x)
        # Norm(x+sublayer(x))
        _x = x # 원본
        x, _ = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = self.norm1(x+_x)

        _x = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.norm2(x + _x)

        return x
    

# %%
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x: 입력 (타겟)
        # enc_output: 인코더 최종 출력
        # tgt_mask: self_attn mask
        # src_mask: 인코더 입력 패딩 무시
        
        # masked self attention
        _x = x
        x, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.dropout(x)
        x = self.norm1(x+_x)

        # cross attention
        _x = x
        x, attn_weights = self.cross_attn(q=x, k=enc_output, v=enc_output, mask=src_mask)
        x = self.dropout(x)
        x = self.norm2(x+_x)

        _x = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.norm3(x+_x)

        return x, attn_weights


# %%
# 레이어 테스트

d_model = 32
n_heads = 4
d_ff = 128
seq_len = 12
batch_size = 2

enc_layer = EncoderLayer(d_model, n_heads, d_ff)
x = torch.randn(batch_size, seq_len, d_model)

enc_out = enc_layer(x)
print(f"Encoder Layer input: {x.shape}")
print(f"Encoder Layer Output: {enc_out.shape}")

dec_layer = DecoderLayer(d_model, n_heads, d_ff)
# 가짜 인코더 출력
memory = torch.randn(batch_size, seq_len, d_model)

dec_out, weights = dec_layer(x, memory, src_mask=None, tgt_mask=None)
print(f"디코더 출력: {dec_out.shape}")
# [2, 12, 32]

if x.shape == enc_out.shape == dec_out.shape:
    print("차원 일치")
else:
    print("차원 불일치")

# %%
def make_pad_mask(q, k, pad_idx):
    # q: [batch, q_len], k: [batch_k_len]
    # k에 있는 패딩 토큰 마스킹 (1: 유효, 0: 패딩)
    
    # [batch, 1, 1, k_len]
    mask = (k != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def make_subsequent_mask(seq_len):
    # 하삼각행렬 1
    # [seq_len, seq_len]
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

    # [1, 1, seq_len, seq_len]
    return mask.unsqueeze(0).unsqueeze(0)

pad_idx = 0
src = torch.tensor([[1, 2, 3, 0, 0]])
pad_mask = make_pad_mask(src, src, pad_idx)
sub_mask = make_subsequent_mask(5)

print(f"pad mask shape: {pad_mask.shape}")
print(pad_mask)
print(f"subsequent mask: {sub_mask[0, 0].int()}")

# %%
class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        # x: [batch, seq_len, d_model]
        # 이미 임베딩, PE 된 x
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x, attn_weights = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x), attn_weights

# %%
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads,
                 d_ff, max_len, pad_idx, dropout=0.1):
        super().__init__()

        self.pad_idx = pad_idx

        # 1. Embedding & PE
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)

        # 2. Encoder & Decoder stacks
        self.encoder = Encoder(d_model, n_layers, n_heads, d_ff, dropout)
        self.decoder = Decoder(d_model, n_layers, n_heads, d_ff, dropout)

        # 3. Final generator
        # d_model -> tgt_vocab_size
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        # src: [batch, src_len], tgt: [batch, tgt_len]

        # 1. mask 생성
        # encoder 
        src_mask = make_pad_mask(src, src, self.pad_idx).to(src.device)

        # Decoder:
        # 패딩 가리기, look-ahead 가리기
        tgt_pad_mask = make_pad_mask(tgt, tgt, self.pad_idx).to(tgt.device)
        tgt_sub_mask = make_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        # cross attention: query는 tgt, key는 src
        src_tgt_mask = make_pad_mask(tgt, src, self.pad_idx).to(src.device)

        # 2. encoder forward
        src_emb = self.dropout(self.pos_encoder(self.src_embedding(src)))
        enc_output = self.encoder.forward(src_emb, src_mask)

        #. decoder forward
        tgt_emb = self.dropout(self.pos_decoder(self.tgt_embedding(tgt)))
        dec_output, attn_weights = self.decoder.forward(tgt_emb, enc_output, src_tgt_mask, tgt_mask)

        logits = self.generator(dec_output)
        return logits, attn_weights

# %%
# 전체 모델 테스트

src_vocab = 100
tgt_vocab = 100
d_model = 32
n_layers = 2
n_heads = 4
d_ff = 64
max_len = 20
pad_idx = 0
model = Transformer(src_vocab, tgt_vocab, d_model, n_layers, n_heads, d_ff, 
max_len, pad_idx)

# 더미 (batch = 2)
# src: 길이 10
src_data = torch.randint(1, src_vocab, (2, 10))
# tgt: 길이 10
tgt_data = torch.randint(1, tgt_vocab, (2, 10))

logits, _ = model(src_data, tgt_data)

print(f"Input src: {src_data.shape}")
print(f"Input tgt: {tgt_data.shape}")
print(f"logits output: {logits.shape}")
print(logits.shape == (2, 10, 100))

# %%



