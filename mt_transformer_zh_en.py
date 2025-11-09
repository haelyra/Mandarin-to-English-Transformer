import math
import csv
from collections import Counter
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def tokenize_en(text: str) -> List[str]:
    return text.lower().strip().split()


try:
    import jieba

    def tokenize_zh(text: str) -> List[str]:
        return [tok for tok in jieba.lcut(text.strip()) if tok.strip()]

except ImportError:
    def tokenize_zh(text: str) -> List[str]:
        return list(text.strip())


class Vocab:
    def __init__(self, min_freq: int = 2, specials=None):
        if specials is None:
            specials = ['<pad>', '<bos>', '<eos>', '<unk>']
        self.min_freq = min_freq
        self.specials = specials
        self.itos: List[str] = []
        self.stoi: dict = {}

    def build(self, token_lists: List[List[str]]):
        counter = Counter()
        for toks in token_lists:
            counter.update(toks)
        self.itos = list(self.specials)
        for tok, f in counter.items():
            if f >= self.min_freq:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @property
    def pad_idx(self) -> int:
        return self.stoi['<pad>']

    @property
    def bos_idx(self) -> int:
        return self.stoi['<bos>']

    @property
    def eos_idx(self) -> int:
        return self.stoi['<eos>']

    @property
    def unk_idx(self) -> int:
        return self.stoi['<unk>']

    def numericalize(
        self,
        toks: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: int = None,
    ) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_idx)
        for t in toks:
            ids.append(self.stoi.get(t, self.unk_idx))
        if add_eos:
            ids.append(self.eos_idx)
        if max_len is not None:
            ids = ids[:max_len]
        return ids


def load_parallel_tsv(
    path: str,
    src_col: str = 'zh',
    tgt_col: str = 'en',
    limit: int = None,
) -> Tuple[List[str], List[str]]:
    """
    Expects a tab-separated file with a header row, including columns src_col and tgt_col.
    """
    src_texts, tgt_texts = [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            s = row[src_col].strip()
            t = row[tgt_col].strip()
            if not s or not t:
                continue
            src_texts.append(s)
            tgt_texts.append(t)
    return src_texts, tgt_texts


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_texts,
        tgt_texts,
        src_tokenizer,
        tgt_tokenizer,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: int = 64,
    ):
        assert len(src_texts) == len(tgt_texts)
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        s = self.src_tok(self.src_texts[idx])
        t = self.tgt_tok(self.tgt_texts[idx])
        s_ids = self.src_vocab.numericalize(s, max_len=self.max_len - 2)
        t_ids = self.tgt_vocab.numericalize(t, max_len=self.max_len - 2)
        return torch.tensor(s_ids, dtype=torch.long), torch.tensor(t_ids, dtype=torch.long)


def pad_sequence(seqs, pad_value: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def collate_fn(batch, src_pad_idx: int, tgt_pad_idx: int):
    src_seqs, tgt_seqs = zip(*batch)
    src = pad_sequence(src_seqs, src_pad_idx)
    tgt = pad_sequence(tgt_seqs, tgt_pad_idx)
    return src, tgt


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class Seq2SeqTransformer(nn.Module):
    """
    Encoder-decoder Transformer with tied positional embeddings,
    using PyTorch's nn.Transformer under the hood.
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.src_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.pos_emb = nn.Embedding(max_len, emb_size)
        self.generator = nn.Linear(emb_size, tgt_vocab_size, bias=False)

        nn.init.xavier_uniform_(self.src_emb.weight)
        nn.init.xavier_uniform_(self.tgt_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        
        self.generator.weight = self.tgt_emb.weight

    def encode(self, src: torch.Tensor, src_mask, src_key_padding_mask):
        src_seq_len = src.size(1)
        pos = torch.arange(0, src_seq_len, device=src.device).unsqueeze(0)
        x = self.src_emb(src) + self.pos_emb(pos)
        x = x.transpose(0, 1)
        return self.transformer.encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
    ):
        tgt_seq_len = tgt.size(1)
        pos = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(0)
        x = self.tgt_emb(tgt) + self.pos_emb(pos)
        x = x.transpose(0, 1)
        return self.transformer.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_key_padding_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
    ):
        memory = self.encode(src, src_mask, src_key_padding_mask)
        outs = self.decode(
            tgt,
            memory,
            tgt_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )
        logits = self.generator(outs)
        return logits.transpose(0, 1)


def create_masks(
    src: torch.Tensor,
    tgt_input: torch.Tensor,
    src_pad_idx: int,
    tgt_pad_idx: int,
    device,
):
    src_key_padding_mask = (src == src_pad_idx)
    tgt_key_padding_mask = (tgt_input == tgt_pad_idx)
    tgt_seq_len = tgt_input.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    return None, tgt_mask, src_key_padding_mask.to(device), tgt_key_padding_mask.to(device)


def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    src_pad_idx,
    tgt_pad_idx,
    device,
):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in tqdm(dataloader, desc="Train", leave=False):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask, src_kpm, tgt_kpm = create_masks(
            src, tgt_input, src_pad_idx, tgt_pad_idx, device
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_kpm,
            tgt_kpm,
            src_kpm,
        )
        logits = logits.reshape(-1, logits.size(-1))
        tgt_out = tgt_out.reshape(-1)

        loss = loss_fn(logits, tgt_out)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        num_tokens = (tgt_out != tgt_pad_idx).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens


@torch.no_grad()
def evaluate_epoch(
    model,
    dataloader,
    loss_fn,
    src_pad_idx,
    tgt_pad_idx,
    device,
):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in tqdm(dataloader, desc="Valid", leave=False):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask, src_kpm, tgt_kpm = create_masks(
            src, tgt_input, src_pad_idx, tgt_pad_idx, device
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_kpm,
            tgt_kpm,
            src_kpm,
        )
        logits = logits.reshape(-1, logits.size(-1))
        tgt_out = tgt_out.reshape(-1)

        loss = loss_fn(logits, tgt_out)

        num_tokens = (tgt_out != tgt_pad_idx).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens


@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    src_pad_idx: int,
    bos_idx: int,
    eos_idx: int,
    max_len: int,
    device,
):
    """
    Greedy decoding for inference: takes (N,T_src) → (N,T_pred)
    """
    model.eval()
    src = src.to(device)
    src_kpm = (src == src_pad_idx).to(device)
    memory = model.encode(src, None, src_kpm)
    batch_size = src.size(0)
    ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model.decode(
            ys,
            memory,
            tgt_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=src_kpm,
        )
        out = out.transpose(0, 1)
        logits = model.generator(out[:, -1, :])
        next_word = logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_word], dim=1)
        if (next_word == eos_idx).all():
            break

    return ys


def ids_to_tokens(ids: List[int], vocab: Vocab) -> List[str]:
    toks = []
    for i in ids:
        tok = vocab.itos[i]
        if tok in ('<bos>', '<eos>', '<pad>'):
            continue
        toks.append(tok)
    return toks


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.tsv")
    parser.add_argument("--valid_path", type=str, default="data/valid.tsv")
    parser.add_argument("--src_col", type=str, default="zh")
    parser.add_argument("--tgt_col", type=str, default="en")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/transformer_zh_en.pt")
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_src, train_tgt = load_parallel_tsv(args.train_path, args.src_col, args.tgt_col)
    valid_src, valid_tgt = load_parallel_tsv(args.valid_path, args.src_col, args.tgt_col)

    src_tokens = [tokenize_zh(s) for s in train_src]
    tgt_tokens = [tokenize_en(t) for t in train_tgt]
    src_vocab = Vocab(min_freq=args.min_freq)
    tgt_vocab = Vocab(min_freq=args.min_freq)
    src_vocab.build(src_tokens)
    tgt_vocab.build(tgt_tokens)
    print(f"Src vocab size: {len(src_vocab.itos)}, Tgt vocab size: {len(tgt_vocab.itos)}")

    train_ds = TranslationDataset(
        train_src,
        train_tgt,
        tokenize_zh,
        tokenize_en,
        src_vocab,
        tgt_vocab,
        max_len=args.max_len,
    )
    valid_ds = TranslationDataset(
        valid_src,
        valid_tgt,
        tokenize_zh,
        tokenize_en,
        src_vocab,
        tgt_vocab,
        max_len=args.max_len,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx, tgt_vocab.pad_idx),
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx, tgt_vocab.pad_idx),
    )

    model = Seq2SeqTransformer(
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        emb_size=args.emb_size,
        nhead=args.heads,
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        dim_feedforward=args.ff_dim,
        dropout=0.1,
        max_len=args.max_len + 5,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.pad_idx,
        label_smoothing=0.1,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.7,
        patience=1,
        verbose=True,
    )

    best_val_loss = float("inf")
    no_improve_epochs = 0

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(
            model,
            train_dl,
            optimizer,
            loss_fn,
            src_vocab.pad_idx,
            tgt_vocab.pad_idx,
            device,
        )
        val_loss = evaluate_epoch(
            model,
            valid_dl,
            loss_fn,
            src_vocab.pad_idx,
            tgt_vocab.pad_idx,
            device,
        )
        print(f"  Train loss: {train_loss:.4f}, ppl: {math.exp(train_loss):.2f}")
        print(f"  Valid loss: {val_loss:.4f}, ppl: {math.exp(val_loss):.2f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve_epochs = 0
            print(f"  Saving checkpoint to {args.checkpoint}")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                    "args": vars(args),
                },
                args.checkpoint,
            )
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print("Early stopping triggered.")
                break

    print("Done.")


if __name__ == "__main__":
    main()
