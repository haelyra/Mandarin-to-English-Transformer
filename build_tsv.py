import json
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RANDOM_SEED = 42
VALID_FRACTION = 0.1


def clean_text(text: str, lang: str):
    """Basic cleaning for both zh and en text."""
    if not text:
        return ""
    
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"``.*?``", " ", text)
    text = re.sub(r"`.*?`", " ", text)

    text = re.sub(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", " ", text)

    text = re.sub(r"\d+", " ", text)

    if lang == "en":
        text = re.sub(r"[^a-zA-Z\s']", " ", text)
        text = text.lower()
    else:
        text = re.sub(r"[^\u4e00-\u9fff。，！？、；：" "'（）—…《》〈〉〈〉·\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_parallel_jsonl(path_zh="data_raw/common_zh_70k.jsonl", path_en="data_raw/common_en_70k.jsonl"):
    pairs = []
    skipped_empty = skipped_short = skipped_long = skipped_turn_mismatch = 0

    print(f"Loading {Path(path_zh).name}...")
    zh_convs = {}
    with open(path_zh, "r", encoding="utf-8") as f_zh:
        for line in tqdm(f_zh, desc="Reading zh file"):
            if not line.strip():
                skipped_empty += 1
                continue
            try:
                ex = json.loads(line)
                conv_id = ex.get("conversation_id")
                if conv_id:
                    zh_convs[conv_id] = ex.get("conversation", [])
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(zh_convs):,} zh conversations")

    print(f"Loading {Path(path_en).name}...")
    en_convs = {}
    with open(path_en, "r", encoding="utf-8") as f_en:
        for line in tqdm(f_en, desc="Reading en file"):
            if not line.strip():
                skipped_empty += 1
                continue
            try:
                ex = json.loads(line)
                conv_id = ex.get("conversation_id")
                if conv_id:
                    en_convs[conv_id] = ex.get("conversation", [])
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(en_convs):,} en conversations")

    common_ids = set(zh_convs.keys()) & set(en_convs.keys())
    print(f"  Found {len(common_ids):,} matching conversation_ids")
    print(f"  Zh-only: {len(zh_convs) - len(common_ids):,}, En-only: {len(en_convs) - len(common_ids):,}")

    print("Extracting sentence pairs...")
    for conv_id in tqdm(common_ids, desc="Processing conversations"):
        conv_zh = zh_convs[conv_id]
        conv_en = en_convs[conv_id]

        if len(conv_zh) != len(conv_en):
            skipped_turn_mismatch += 1
            continue

        for turn_zh, turn_en in zip(conv_zh, conv_en):
            for role in ("human", "assistant"):
                zh = clean_text((turn_zh.get(role) or ""), lang="zh")
                en = clean_text((turn_en.get(role) or ""), lang="en")

                if len(zh) < 3 or len(en) < 3:
                    skipped_short += 1
                    continue

                if len(zh) > 1024 or len(en) > 1024:
                    skipped_long += 1
                    continue

                pairs.append({"zh": zh, "en": en})

    print(f"\nExtraction summary:")
    print(f"  Skipped empty lines: {skipped_empty:,}")
    print(f"  Skipped turn count mismatch: {skipped_turn_mismatch:,}")
    print(f"  Skipped short (<3 chars): {skipped_short:,}")
    print(f"  Skipped long (>1024 chars): {skipped_long:,}")
    print(f"  Total pairs extracted: {len(pairs):,}")

    return pairs


def main():
    random.seed(RANDOM_SEED)

    raw_dir = Path("data_raw")
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    common_pairs = load_parallel_jsonl(
        raw_dir / "common_zh_70k.jsonl",
        raw_dir / "common_en_70k.jsonl",
    )

    comp_zh = raw_dir / "computer_zh_26k(fixed).jsonl"
    comp_en = raw_dir / "computer_en_26k(fixed).jsonl"
    if comp_zh.exists() and comp_en.exists():
        comp_pairs = load_parallel_jsonl(comp_zh, comp_en)
    else:
        comp_pairs = []

    all_pairs = common_pairs + comp_pairs
    print(f"Total raw pairs: {len(all_pairs):,}")

    df = pd.DataFrame(all_pairs).drop_duplicates().reset_index(drop=True)
    print(f"After dedup: {len(df):,}")

    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    n_total = len(df)
    n_valid = int(n_total * VALID_FRACTION)

    valid_df = df.iloc[:n_valid].reset_index(drop=True)
    train_df = df.iloc[n_valid:].reset_index(drop=True)

    print(f"Train size: {len(train_df):,}")
    print(f"Valid size: {len(valid_df):,}")

    train_df.to_csv(out_dir / "train.tsv", sep="\t", index=False)
    valid_df.to_csv(out_dir / "valid.tsv", sep="\t", index=False)

    print(f"Wrote {out_dir/'train.tsv'} and {out_dir/'valid.tsv'}")


if __name__ == "__main__":
    main()
