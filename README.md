# Mandarin-English Neural Machine Translation

A PyTorch implementation of a Transformer-based neural machine translation program for translating between Mandarin and English. This project implements the encoder/decoder architecture from "Attention Is All You Need" (Vaswani et al., 2017) from scratch.

# Features: 
- Data cleaning and alignment from JSONL to TSV  
- Vocabulary creation with frequency filtering  
- Full transformer encoder/decoder (multi-head attention, FFN, positional encoding)  
- Checkpoints
- Training with validation and greedy decoding

## Dataset

Uses the [ShareGPT Chinese-English dataset](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) with about 64k training and 7k validation pairs.

## Model Details
- 3 encoder & decoder layers
- 8 attention heads
- 256-dim embeddings
- Batch size: 32
- Learning rate: 3e-4
- Max sequence length: 64

## Results
- **12.2M parameters** trained from scratch
- **Final perplexity: 26.50** (train), **40.26** (validation)
- **Source vocabulary: 13,940** tokens
- **Target vocabulary: 18,353** tokens
- Trained on **21,500** parallel sentence pairs
- Achieved **~96% reduction** in perplexity over 10 epochs (675 → 26.5)

(Supports CPU and GPU)