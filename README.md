# Baby GPT on Shakespeare

*A learning experiment: training a small character-level GPT from scratch.*

I'm trying to understand how GPTs work by training a tiny one myself. This repo follows Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT license), and these are my notes from running it. None of this is original work. It's just a record of what I tried and what I learned.

## Setup

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

What each package is for, as far as I understand it:

- `torch`: the deep learning framework the model is built in
- `numpy`: array handling
- `transformers`: loading GPT-2 checkpoints (not needed for the Shakespeare run)
- `datasets`: downloading and preprocessing OpenWebText (also not needed here)
- `tiktoken`: OpenAI's fast BPE tokenizer
- `wandb`: optional experiment logging
- `tqdm`: progress bars

## Experiment 1: Shakespeare, character by character

I started with the smallest setup: a character-level model trained on Shakespeare's works. The data prep script downloads a ~1MB text file and turns it into one long stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates `train.bin` and `val.bin`. Then I trained with the config in [config/train_shakespeare_char.py](config/train_shakespeare_char.py):

```sh
python train.py config/train_shakespeare_char.py
```

The model has a context of 256 characters, 384 feature channels, and 6 Transformer layers with 6 attention heads each.

**My result:** [your hardware, training time, best validation loss]
*(For reference, nanoGPT reports about 3 minutes and a validation loss of 1.47 on one A100.)*

Checkpoints are saved to `out-shakespeare-char`, and I sampled from the best one with:

```sh
python sample.py --out_dir=out-shakespeare-char
```

**A sample from my run:**

```
[paste a real sample here]
```

## Experiment 2: scaling it down for a laptop

I don't have a GPU, so I shrank everything to run on a CPU:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

What I changed and why, as I understand it:

- `--device=cpu --compile=False`: no GPU, and PyTorch 2.0 compile is turned off
- `--eval_iters=20` (down from 200): noisier but faster validation
- `--block_size=64` (down from 256): shorter context
- `--batch_size=12` (down from 64): fewer examples per step
- `--n_layer=4 --n_head=4 --n_embd=128`: a much smaller Transformer
- `--max_iters=2000 --lr_decay_iters=2000`: fewer steps, with the learning rate decaying over the same length
- `--dropout=0.0`: less regularization, since the network is already tiny

**My result:** [training time, final loss]
*(nanoGPT reports about 3 minutes and a loss of 1.88 for this setup.)*

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

```
[paste a real sample here]
```

On an Apple Silicon Mac, `--device=mps` uses the on-chip GPU and should speed training up noticeably.

## What I learned

- [e.g. what changed when I made the model smaller]
- [e.g. how loss related to sample quality]
- [e.g. what surprised me]

## What I want to try next

- [e.g. a subword tokenizer (BPE) instead of characters]
- [e.g. a different dataset]
