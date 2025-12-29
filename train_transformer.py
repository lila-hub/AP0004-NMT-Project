# ===== train_transformer.py =====
import math, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sentencepiece as spm
import sacrebleu

PROJECT = Path(r"E:\NLPpro\nmt_project")
DATA = PROJECT / "data_txt"
SPM_DIR = PROJECT / "spm"
SPM_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ---------- SentencePiece ----------
def train_spm():
    corpus = SPM_DIR / "corpus.txt"
    if not corpus.exists():
        with open(corpus, "w", encoding="utf-8") as f:
            f.write((DATA / "train.zh").read_text(encoding="utf-8"))
            f.write((DATA / "train.en").read_text(encoding="utf-8"))

    if not (SPM_DIR / "spm.model").exists():
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(SPM_DIR / "spm"),
            vocab_size=8000,
            model_type="bpe",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )

train_spm()
sp = spm.SentencePieceProcessor(model_file=str(SPM_DIR / "spm.model"))

PAD, UNK, BOS, EOS = 0, 1, 2, 3
VOCAB = sp.vocab_size()

def encode(line):
    return [BOS] + sp.encode(line.strip()) + [EOS]

class ParallelDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src.read_text(encoding="utf-8").splitlines()
        self.tgt = tgt.read_text(encoding="utf-8").splitlines()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (
            torch.tensor(encode(self.src[idx])),
            torch.tensor(encode(self.tgt[idx]))
        )

def collate(batch):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=PAD)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD)
    return src, tgt

class TransformerNMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 256, padding_idx=PAD)
        self.pos = nn.Embedding(512, 256)
        self.tf = nn.Transformer(
            d_model=256, nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            batch_first=True
        )
        self.out = nn.Linear(256, VOCAB)

    def forward(self, src, tgt):
        B, S = src.shape
        T = tgt.shape[1]
        pos_s = torch.arange(S, device=src.device)
        pos_t = torch.arange(T, device=tgt.device)

        src = self.emb(src) + self.pos(pos_s)
        tgt = self.emb(tgt) + self.pos(pos_t)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(src.device)
        h = self.tf(src, tgt, tgt_mask=tgt_mask)
        return self.out(h)

def main():
    train_ds = ParallelDataset(DATA / "train.zh", DATA / "train.en")
    valid_ds = ParallelDataset(DATA / "valid.zh", DATA / "valid.en")

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=32, collate_fn=collate)

    model = TransformerNMT().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    for ep in range(3):
        model.train()
        for src, tgt in tqdm(train_dl, desc=f"Epoch {ep+1}"):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(
                logits.reshape(-1, VOCAB),
                tgt[:, 1:].reshape(-1)
            )
            loss.backward()
            opt.step()

        model.eval()
        hyps, refs = [], []
        with torch.no_grad():
            for src, tgt in valid_dl:
                src = src.to(DEVICE)
                pred = model(src, tgt[:, :-1].to(DEVICE)).argmax(-1).cpu()
                for i in range(pred.size(0)):
                    hyps.append(sp.decode(pred[i].tolist()))
                    refs.append(sp.decode(tgt[i, 1:].tolist()))

        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        print(f"Epoch {ep+1} BLEU: {bleu:.2f}")

if __name__ == "__main__":
    main()
