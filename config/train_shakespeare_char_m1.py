# train a miniature character-level shakespeare model on a MacBook (M1, 8GB)
# based on nanoGPT's config/train_shakespeare_char.py, scaled down to fit a laptop
# [mine] every value I changed is marked below

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 20 # [mine] down from 200: noisier but faster validation
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-m1' # [mine]

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 12 # [mine] down from 64
block_size = 64 # [mine] context of up to 64 previous characters, down from 256

# baby GPT model :)
n_layer = 4 # [mine] down from 6
n_head = 4 # [mine] down from 6
n_embd = 128 # [mine] down from 384
dropout = 0.0 # [mine] down from 0.2, the network is already tiny

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000 # [mine] down from 5000
lr_decay_iters = 2000 # [mine] make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# [mine] macbook settings, now built in
device = 'mps' # M1 on-chip GPU. If mps gives errors, change to 'cpu'
compile = False # do not torch compile the model
