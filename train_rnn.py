# ===== train_rnn.py =====
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import sentencepiece as spm
import sacrebleu

PROJECT = Path(r"E:\NLPpro\nmt_project")
DATA = PROJECT / "data_txt"
SPM_DIR = PROJECT / "spm"   # 复用你 Transformer 训练时生成的 spm.model / spm.vocab

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# -------- load SentencePiece --------
sp = spm.SentencePieceProcessor(model_file=str(SPM_DIR / "spm.model"))
PAD, UNK, BOS, EOS = 0, 1, 2, 3
VOCAB = sp.vocab_size()

def encode(line):
    return [BOS] + sp.encode(line.strip()) + [EOS]

def decode(ids):
    ids = [i for i in ids if i not in (PAD, BOS, EOS)]
    return sp.decode(ids)

# -------- Dataset --------
class ParallelDataset(Dataset):
    def __init__(self, src, tgt, max_len=128):
        self.src = src.read_text(encoding="utf-8").splitlines()
        self.tgt = tgt.read_text(encoding="utf-8").splitlines()
        self.max_len = max_len
        assert len(self.src) == len(self.tgt)

    def __len__(self): return len(self.src)

    def __getitem__(self, idx):
        s = encode(self.src[idx])[:self.max_len]
        t = encode(self.tgt[idx])[:self.max_len]
        return torch.tensor(s), torch.tensor(t)

def collate(batch):
    src, tgt = zip(*batch)
    src_lens = torch.tensor([len(x) for x in src], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgt], dtype=torch.long)
    src = pad_sequence(src, batch_first=True, padding_value=PAD)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD)
    return src, src_lens, tgt, tgt_lens

# -------- RNN + Attention --------
class Encoder(nn.Module):
    def __init__(self, emb_dim=256, hid=512, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(
            emb_dim, hid, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0, bidirectional=True
        )
        self.proj = nn.Linear(hid * 2, hid)  # to match decoder hid

    def forward(self, src, src_lens):
        x = self.emb(src)
        packed = pack_padded_sequence(x, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, h = self.rnn(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)  # [B,S,2H]
        out = self.proj(out)  # [B,S,H]
        return out

class BahdanauAttention(nn.Module):
    def __init__(self, hid=512):
        super().__init__()
        self.Wq = nn.Linear(hid, hid, bias=False)
        self.Wk = nn.Linear(hid, hid, bias=False)
        self.v  = nn.Linear(hid, 1, bias=False)

    def forward(self, query, keys, key_mask):
        # query: [B,H], keys: [B,S,H], key_mask: [B,S] (True=pad)
        q = self.Wq(query).unsqueeze(1)      # [B,1,H]
        k = self.Wk(keys)                    # [B,S,H]
        e = self.v(torch.tanh(q + k)).squeeze(-1)  # [B,S]
        e = e.masked_fill(key_mask, -1e9)
        a = torch.softmax(e, dim=-1)         # [B,S]
        ctx = torch.bmm(a.unsqueeze(1), keys).squeeze(1)  # [B,H]
        return ctx, a

class Decoder(nn.Module):
    def __init__(self, emb_dim=256, hid=512, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb_dim, padding_idx=PAD)
        self.rnn = nn.GRU(
            emb_dim + hid, hid, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.attn = BahdanauAttention(hid)
        self.out = nn.Linear(hid * 2, VOCAB)

    def forward(self, enc_out, src_mask, tgt_in):
        # teacher forcing: feed gold tokens
        B, T = tgt_in.size()
        hid = torch.zeros(2, B, 512, device=tgt_in.device)  # layers=2

        logits = []
        ctx = torch.zeros(B, 512, device=tgt_in.device)

        emb = self.emb(tgt_in)  # [B,T,E]
        for t in range(T):
            inp = torch.cat([emb[:, t, :], ctx], dim=-1).unsqueeze(1)  # [B,1,E+H]
            out, hid = self.rnn(inp, hid)  # out: [B,1,H]
            dec = out.squeeze(1)           # [B,H]
            ctx, _ = self.attn(dec, enc_out, src_mask)  # [B,H]
            logit = self.out(torch.cat([dec, ctx], dim=-1))  # [B,V]
            logits.append(logit.unsqueeze(1))
        return torch.cat(logits, dim=1)  # [B,T,V]

@torch.no_grad()
def greedy_decode(enc, dec, src, src_lens, max_len=128):
    enc.eval(); dec.eval()
    enc_out = enc(src, src_lens)
    src_mask = (src == PAD)

    B = src.size(0)
    ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
    hid = torch.zeros(2, B, 512, device=src.device)
    ctx = torch.zeros(B, 512, device=src.device)

    for _ in range(max_len - 1):
        emb = dec.emb(ys[:, -1])  # [B,E]
        inp = torch.cat([emb, ctx], dim=-1).unsqueeze(1)
        out, hid = dec.rnn(inp, hid)
        dec_h = out.squeeze(1)
        ctx, _ = dec.attn(dec_h, enc_out, src_mask)
        logit = dec.out(torch.cat([dec_h, ctx], dim=-1))
        next_id = logit.argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_id], dim=1)
        if (next_id == EOS).all():
            break
    return ys

def main():
    train_ds = ParallelDataset(DATA / "train.zh", DATA / "train.en", max_len=128)
    valid_ds = ParallelDataset(DATA / "valid.zh", DATA / "valid.en", max_len=128)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False, collate_fn=collate)

    enc = Encoder().to(DEVICE)
    dec = Decoder().to(DEVICE)

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    EPOCHS = 3
    for ep in range(1, EPOCHS + 1):
        enc.train(); dec.train()
        pbar = tqdm(train_dl, desc=f"RNN Epoch {ep}")
        for src, src_lens, tgt, tgt_lens in pbar:
            src, src_lens, tgt = src.to(DEVICE), src_lens.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()

            enc_out = enc(src, src_lens)
            src_mask = (src == PAD)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = dec(enc_out, src_mask, tgt_in)

            loss = loss_fn(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        # valid BLEU (greedy)
        enc.eval(); dec.eval()
        hyps, refs = [], []
        with torch.no_grad():
            for src, src_lens, tgt, tgt_lens in valid_dl:
                src, src_lens = src.to(DEVICE), src_lens.to(DEVICE)
                pred = greedy_decode(enc, dec, src, src_lens, max_len=128).cpu()
                for i in range(pred.size(0)):
                    hyps.append(decode(pred[i].tolist()))
                for i in range(tgt.size(0)):
                    refs.append(decode(tgt[i].tolist()))
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        print(f"RNN Epoch {ep} BLEU: {bleu:.2f}")

    # 保存 demo
    demo_path = PROJECT / "demo_rnn.txt"
    with open(demo_path, "w", encoding="utf-8") as f:
        for k in range(10):
            zh = train_ds.src[k]
            en = train_ds.tgt[k]
            f.write(f"ZH: {zh}\nGT: {en}\n\n")
    print("Saved demo ->", demo_path)

if __name__ == "__main__":
    main()
