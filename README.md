# Baby GPT on Shakespeare

*A learning experiment: training a small character-level GPT from scratch, on a MacBook.*

I'm trying to understand how GPTs work by training a tiny one myself. I only have a MacBook (M1, 8GB), so everything here is scaled to run on it. This repo follows Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT license), and these are my notes from working through it. The base code is his; the small changes I made are listed below.

## Setup

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib pandas
```

What each package is for, as far as I understand it:

- `torch`: the deep learning framework the model is built in
- `numpy`: array handling
- `transformers`: loading GPT-2 checkpoints (not needed for the Shakespeare run)
- `datasets`: downloading and preprocessing OpenWebText (also not needed here)
- `tiktoken`: OpenAI's fast BPE tokenizer
- `wandb`: optional experiment logging
- `tqdm`: progress bars
- `matplotlib`, `pandas`: plotting the loss curve from my log

## Step 1: prepare the data

The script downloads a ~1MB text file of Shakespeare's works and turns it into one long stream of integers, one per character:

```sh
python data/shakespeare_char/prepare.py
```

This creates `train.bin` and `val.bin`.

## Step 2: train a small model on the MacBook

nanoGPT's default config assumes a GPU, so I shrank the model to fit a laptop:

```sh
python train.py config/train_shakespeare_char_m1.py --device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

What each flag does, as I understand it:

- `--device=mps`: run on the Mac's on-chip GPU
- `--compile=False`: PyTorch 2.0 compile turned off
- `--eval_iters=20` (down from 200): noisier but faster validation
- `--block_size=64` (down from 256): the model sees 64 characters of context instead of 256
- `--batch_size=12` (down from 64): fewer examples per step
- `--n_layer=4 --n_head=4 --n_embd=128`: 4 layers, 4 attention heads, 128 feature channels, instead of 6, 6 and 384
- `--max_iters=2000 --lr_decay_iters=2000`: fewer steps, with the learning rate decaying over the same length
- `--dropout=0.0`: less regularization, since the network is already tiny

To check that everything works before a full run, try a short one first by adding `--max_iters=100 --eval_interval=50`.

### Using the Mac's GPU

I'm on an M1 with 8GB of unified memory, so I train with `--device=mps`, which runs on the on-chip GPU through Metal Performance Shaders. nanoGPT says this can speed training up 2-3x compared to the CPU. If `mps` gives errors, replace it with `--device=cpu` and everything else stays the same.

Since the 8GB is shared between the system and the GPU, this small model fits comfortably. When scaling up, I'll change one setting at a time (for example `--n_embd` or `--block_size`) and close heavy apps before longer runs.

## Step 3: look at the loss curve

Training writes every evaluation (iteration, train loss, validation loss, learning rate, elapsed seconds) to `out-shakespeare-char/loss_log.csv`. To plot it:

```sh
python plot_loss.py
```

This saves `loss_curve.png` in the repo root. The CSV only gets a row every `eval_interval` steps, so for a smoother curve add `--eval_interval=100` to the training command (each evaluation costs a bit of time).

## Step 4: sample from the model

```sh
python sample.py --out_dir=out-shakespeare-char --device=mps
```

nanoGPT shows output like this for the small laptop setup (not from my run):

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

## What I changed from nanoGPT

- `train.py`: logs train loss, validation loss, learning rate, and elapsed time to `loss_log.csv` at every evaluation. Also defaults `dtype` to float32 when there is no CUDA GPU, and only enables the GradScaler for float16 on CUDA, so `mps` and CPU runs don't trigger the CUDA warning.
- `plot_loss.py` (new): plots the loss curve from the CSV.

Everything else is unchanged from the original.

## For comparison: the full-size run

nanoGPT's default config trains a bigger model: 256 characters of context, 384 feature channels, and 6 layers with 6 heads each. It reports about 3 minutes on one A100 GPU and a best validation loss of 1.4697. I haven't run this one, since it needs a GPU I don't have:

```sh
python train.py config/train_shakespeare_char.py
```

## What the numbers show

- nanoGPT reports about 1.88 loss for the small laptop setup versus 1.47 for the full-size one. Shrinking the model, context, and batch size has a visible cost in quality.
- The samples reflect that: the small model picks up the shape of words and verse, but not their meaning.
- Character-level modeling keeps the setup simple (a vocabulary of about 65 symbols), but the model has to learn spelling from scratch and only sees a very short window of text.

## What I want to try next

- Sample from the pretrained GPT-2 (124M) and compare it with my character-level model
- Finetune GPT-2 (124M) on Shakespeare using the BPE tokenizer, with a smaller batch size and context to fit in 8GB
- Change one setting at a time to see which one affects the loss most
- Try a different dataset

## License

MIT, same as nanoGPT. See [LICENSE](LICENSE).
