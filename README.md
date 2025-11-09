# Mandarin-English Neural Machine Translation

A PyTorch implementation of a Transformer-based neural machine translation program for translating between Mandarin and English. This project and implements the encoder/decoder architecture from "Attention Is All You Need" (Vaswani et al., 2017) from scratch.

# Features: 
- Data cleaning and alignment from JSONL to TSV  
- Vocabulary creation with frequency filtering  
- Full transformer encoder/decoder (multi-head attention, FFN, positional encoding)  
- Checkpoints
- Training with validation and greedy decoding

## Dataset

Uses the [ShareGPT Chinese-English dataset](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) with about 64k training and 7k validation pairs.

Model Details
- 3 encoder & decoder layers
- 8 attention heads
- 256-dim embeddings
- Batch size: 32
- Learning rate: 1e-4
- Max sequence length: 64


(Supports CPU and GPU)