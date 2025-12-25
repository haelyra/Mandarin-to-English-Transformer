# Mandarin-English Neural Machine Translation

Haley Chen. (2025). haelyra/Mandarin-to-English-Transformer: 
v1.1 (v1.1). Zenodo. [https://doi.org/10.5281/zenodo.18057192](https://doi.org/10.5281/zenodo.18057228)

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
- **15.4M parameters** trained from scratch
- **Final perplexity: 15.41** (train), **20.36** (validation)
- **Source vocabulary: 19,098** tokens
- **Target vocabulary: 25,515** tokens
- Trained on **43,000** parallel sentence pairs
- Achieved **~96.8% reduction** in train perplexity over 10 epochs (488.12 → 15.41)

(Supports CPU and GPU)
