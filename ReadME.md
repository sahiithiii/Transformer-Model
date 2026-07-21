# Transformer Model from Scratch

A complete PyTorch implementation of the Transformer architecture proposed in the paper **"Attention Is All You Need"** by Vaswani et al. This repository focuses on building the Transformer from first principles, implementing every major component without relying on high-level libraries such as `torch.nn.Transformer`.

The project is intended for learning the internal workings of Transformers, including self-attention, positional encoding, multi-head attention, encoder-decoder architecture, masking, and training.

## Overview

This implementation follows the original research paper as closely as possible and provides a modular codebase where each component is implemented independently. The objective is to understand how Transformer models work internally rather than using pre-built implementations.

The repository includes:

- Input Embeddings
- Positional Encoding
- Layer Normalization
- Residual Connections
- Multi-Head Self Attention
- Feed Forward Networks
- Encoder
- Decoder
- Encoder-Decoder Transformer
- Source and Target Masks
- Training Pipeline
- Greedy Decoding for Inference

## Architecture

The implemented architecture consists of:

- Input Embedding Layer
- Positional Encoding
- N Encoder Blocks
- N Decoder Blocks
- Multi-Head Attention
- Position-wise Feed Forward Networks
- Layer Normalization
- Residual Connections
- Final Linear Projection to Vocabulary

The implementation closely follows the architecture presented in the original paper.

## Repository Structure

```
Transformer-Model/
│
├── config.py                 # Training configuration
├── dataset.py                # Dataset loading and preprocessing
├── model.py                  # Transformer architecture implementation
├── train.py                  # Training script
├── translate.py              # Inference and translation
├── tokenizer_*.json          # Tokenizers
├── weights/                  # Saved model checkpoints
└── README.md
```

## Implemented Components

### Embedding Layer

Converts token indices into dense vector representations and scales them by √d_model.

### Positional Encoding

Implements sinusoidal positional encodings to provide positional information without recurrence.

### Multi-Head Attention

Implements:

- Query
- Key
- Value projections
- Scaled Dot-Product Attention
- Multi-Head concatenation
- Output projection

### Feed Forward Network

Each Transformer block contains a position-wise feed-forward network consisting of two linear layers with ReLU activation.

### Residual Connections

Residual connections are applied around every attention and feed-forward block, followed by Layer Normalization.

### Encoder

Each encoder layer contains:

- Multi-Head Self Attention
- Feed Forward Network
- Layer Normalization
- Residual Connections

### Decoder

Each decoder layer contains:

- Masked Multi-Head Self Attention
- Cross Attention
- Feed Forward Network
- Layer Normalization
- Residual Connections

### Output Projection

Projects decoder outputs to vocabulary size for token prediction.

## Training

The training pipeline includes:

- Dataset preparation
- Tokenization
- Padding and masking
- Teacher forcing
- Cross Entropy Loss
- Adam Optimizer
- Checkpoint saving
- Validation after each epoch

Run training using:

```bash
python train.py
```

## Inference

Generate translations using:

```bash
python model.py
```

The decoder performs autoregressive generation using greedy decoding.

## Technologies Used

- Python
- PyTorch
- TorchText
- Hugging Face Tokenizers

## Learning Objectives

This project was developed to gain a deep understanding of:

- Transformer architecture
- Self-attention mechanism
- Multi-head attention
- Encoder-decoder models
- Sequence-to-sequence learning
- Positional encoding
- Training neural machine translation models
- PyTorch model implementation from scratch

## Reference

**Attention Is All You Need**
Paper: https://arxiv.org/abs/1706.03762

## Acknowledgements

This implementation is based on the concepts introduced in the paper *Attention Is All You Need* and was developed as an educational project to understand Transformer architecture at a low level by implementing every component from scratch.
