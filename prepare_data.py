import json
from pathlib import Path

# 改成你的数据集路径
DATA_DIR = Path(r"E:\NLPpro\nmt_project\data")
OUT_DIR = Path(r"E:\NLPpro\nmt_project\data_txt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def convert(split_in, zh_out, en_out):
    with open(split_in, "r", encoding="utf-8") as fin, \
         open(zh_out, "w", encoding="utf-8") as fzh, \
         open(en_out, "w", encoding="utf-8") as fen:
        for line in fin:
            obj = json.loads(line)
            zh = obj.get("zh", "").strip()
            en = obj.get("en", "").strip()
            if zh and en:
                fzh.write(zh + "\n")
                fen.write(en + "\n")

convert(DATA_DIR / "train_10k.jsonl", OUT_DIR / "train.zh", OUT_DIR / "train.en")
convert(DATA_DIR / "valid.jsonl",     OUT_DIR / "valid.zh", OUT_DIR / "valid.en")
convert(DATA_DIR / "test.jsonl",      OUT_DIR / "test.zh",  OUT_DIR / "test.en")

print("Done. Saved to:", OUT_DIR)
