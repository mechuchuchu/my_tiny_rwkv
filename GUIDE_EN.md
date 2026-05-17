# RWKV-7 `train_temp` — Detailed Usage Guide (English)
# 2026-5-17
This guide explains how to train an RWKV-7 ("Goose", x070) model from scratch using the code under `RWKV-v7/train_temp/`. It covers the full pipeline:

1. Environment setup
2. Folder/file layout of `train_temp`
3. Preparing the dataset (`.bin` / `.idx`) — using the v5 `make_data.py`
4. Computing `magic_prime`
5. Stage 1 — generating the initial weights (`demo-training-prepare.sh`)
6. Stage 3 — running real training (`demo-training-run.sh`)
7. Resuming a stopped run / fine-tuning from a base model
8. Important CLI flags (line-by-line)
9. Multi-GPU / multi-node
10. VRAM tuning (`grad_cp`, `head_chunk`, `micro_bsz`, kernel selection)
11. Logs, checkpoints, and `wandb`
12. Common pitfalls

---

## 1. Environment

Tested combinations (see `requirements.txt`):

- Python **3.10+**
- PyTorch **2.5+** with CUDA **12.x** (any current cu12 build works — not pinned to cu121)
- DeepSpeed (latest)
- `pytorch-lightning==1.9.5` ← **must be exactly this version**
- `ninja`, `wandb` (optional)

```bash
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade
```

The default config trains on a single GPU with ~7 GB VRAM (L12-D768, ctx 512, micro_bsz 16). Reduce `micro_bsz` or enable `grad_cp=1` / `head_chunk=4096` if you have less.

The CUDA kernels under `cuda/` are JIT-compiled on first run via `torch.utils.cpp_extension.load`. If the launch hangs at "Loading extension…", remove the lock files in `TORCH_EXTENSIONS_DIR` (default `~/.cache/torch_extensions/`) and retry.

---

## 2. Folder layout

```
RWKV-v7/train_temp/
├── README.md                         # original short readme
├── requirements.txt
├── train.py                          # entry point (argparse + Lightning Trainer)
├── rwkv7_train_simplified.py         # simplified single-file reference (read-only study)
├── demo-training-prepare.sh          # Stage-1 script: generate rwkv-init.pth (MiniPile)
├── demo-training-run.sh              # Stage-3 script: actual training (MiniPile)
├── demo-training-prepare-v7-pile.sh  # same but for full Pile (50304 vocab, ctx 4096)
├── demo-training-run-v7-pile.sh
├── src/
│   ├── model.py     # RWKV-7 model (init schedule, optimizer groups, layers)
│   ├── trainer.py   # Lightning callback: LR schedule, checkpoint save, logging
│   ├── dataset.py   # MyDataset: shuffled binidx sampler using magic_prime
│   └── binidx.py    # MMapIndexedDataset reader (same format as Megatron)
└── cuda/            # WKV-7 / cmix / tmix / clampw / head-l2wrap CUDA kernels
```

Outputs go to `out/L<N_LAYER>-D<N_EMBD>-<MODEL_TYPE>/`:

- `rwkv-init.pth` — initial weights (Stage 1)
- `rwkv-<N>.pth` — checkpoint at miniepoch N (Stage 3, every `--epoch_save`)
- `rwkv-final.pth` — final checkpoint when `my_exit_tokens` is reached
- `train_log.txt` — per-epoch loss / perplexity / LR / timestamp

---

## 3. Preparing the dataset (`.bin` / `.idx`)

The trainer only accepts `--data_type binidx` (Megatron-style memory-mapped index). You point `--data_file` at the **prefix** (no extension), and it reads `<prefix>.bin` and `<prefix>.idx`.

### 3a. Option A — download a pre-tokenized dataset (MiniPile, easiest)

```bash
cd RWKV-v7/train_temp
mkdir -p data
wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

This yields `1 498 226 207` tokens with vocab 65536 (RWKV world tokenizer `rwkv_vocab_v20230424.txt`). It matches the default in `demo-training-prepare.sh` exactly.

### 3b. Option B — build `.bin` / `.idx` from your own JSONL (using v5 `make_data.py`)

`train_temp` does not ship its own tokenizer/builder; the canonical tool lives in `RWKV-v5/make_data.py`. It uses the RWKV world tokenizer and writes the same MMapIndexedDataset format that `src/binidx.py` reads.

Required JSONL format (one document per line, **no empty lines**):

```
{"text": "first document ..."}
{"text": "second document ..."}
```

Run:

```bash
cd /home/mechu/rwkv-lm/RWKV-v5
python make_data.py path/to/your.jsonl <N_EPOCH> <CTX_LEN>
```

- `<N_EPOCH>` — how many shuffled passes of the data to concatenate. For pretraining, `1` is fine if you have lots of data; for finetuning small jsonl, use `3` or more so the random sampler in `dataset.py` sees enough material.
- `<CTX_LEN>` — used only to compute and print a matching `magic_prime`.

What it does, end-to-end:

1. Reads `your.jsonl`, strips empty lines.
2. Shuffles and concatenates the lines `N_EPOCH` times into `make_data_temp.jsonl`.
3. Tokenizes each `"text"` with `TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")` and appends token `0` as end-of-doc.
4. Writes `your.bin` and `your.idx` in the current directory.
5. Round-trips a sample to verify tokenizer correctness.
6. Prints the largest `3n+2` prime ≤ `tokens/CTX_LEN - 1` as **`magic_prime`**, along with the exact `--my_exit_tokens` / `--magic_prime` / `--ctx_len` flags to paste into your shell scripts.

Output goes to the current directory, so run it inside the folder where you want the files (or move them afterwards). Copy `your.bin` and `your.idx` into `RWKV-v7/train_temp/data/`, then set `--data_file "data/your"` in both shell scripts.

**Vocab size:** the world tokenizer outputs token ids up to ~65535, so use `--vocab_size 65536` (same as MiniPile). If you tokenized with the GPT-NeoX/Pile tokenizer instead, use `--vocab_size 50304` (and see `demo-training-prepare-v7-pile.sh`).

### 3c. Option C — non-RWKV tokenizer

The format itself is just MMapIndexedDataset (Megatron). Anything that emits `.bin` / `.idx` in that schema (e.g., the Megatron-DeepSpeed preprocessor) will work. Make sure the dtype is one of those declared in `src/binidx.py` (`uint16` is what `make_data.py` uses — fine up to vocab 65 535).

---

## 4. Computing `magic_prime`

The dataset sampler in `src/dataset.py` deterministically maps step index → file offset using:

```
ii  = 1 + epoch * 40320 + idx * world_size + rank
i   = ((⌊magic_prime · (√5 − 1)/2⌋ · ii³) mod magic_prime) · ctx_len
```

For this to visit every chunk roughly once per "epoch" without collisions, `magic_prime` must be:

- a **prime number**,
- satisfying **`magic_prime % 3 == 2`** (i.e., the largest `3n+2` prime fitting),
- with **`0.9 < magic_prime / (data_size // ctx_len) ≤ 1.0`**.

These three assertions live in `MyDataset.__init__` (`src/dataset.py:42-44`) — if any fails, training aborts immediately.

### 4a. Easy way — let `make_data.py` print it

If you built `.bin/.idx` with `RWKV-v5/make_data.py`, it already printed the line:

```
--my_exit_tokens <total_tokens> --magic_prime <N> --ctx_len <CTX_LEN>
```

Paste it verbatim.

### 4b. Existing `.bin/.idx` — use `compute_magic_prime.py`

`RWKV-v5/compute_magic_prime.py` does only step 6 above:

```bash
cd /home/mechu/rwkv-lm/RWKV-v5
# edit DATA_NAME and CTX_LEN at the top of the file
python compute_magic_prime.py
```

It prints `magic_prime` and the exact CLI flags.

### 4c. Manual / web

`magic_prime` is the largest prime ≤ `data_size/ctx_len − 1` with `prime % 3 == 2`. For the default MiniPile + ctx 512:

```
data_size/ctx_len − 1 = 1 498 226 207 / 512 − 1 ≈ 2 926 222.06
→ largest 3n+2 prime ≤ that = 2 926 181
```

You can use https://www.dcode.fr/prime-numbers-search to search downward. **You must re-compute `magic_prime` every time `ctx_len` changes.**

---

## 5. Stage 1 — generate `rwkv-init.pth`

`demo-training-prepare.sh` runs `train.py` with `--train_stage 1`, which calls `generate_init_weight` in `src/trainer.py`. That function uses the per-tensor init schedule defined in `model.generate_init_weight()` (see the table in the original README), writes `rwkv-init.pth` to `PROJ_DIR`, and exits.

Edit the top of `demo-training-prepare.sh` to set:

| Variable      | Meaning                                                                                |
|---------------|----------------------------------------------------------------------------------------|
| `MODEL_TYPE`  | `x070` for RWKV-7. (Don't change unless you're testing a variant.)                     |
| `N_LAYER`     | depth                                                                                  |
| `N_EMBD`      | width (= `dim_att`; FFN dim defaults to `3.5×N_EMBD` rounded to /32)                   |
| `CTX_LEN`     | training context length. **Must match the `magic_prime` you computed.**                |
| `PROJ_DIR`    | output folder. The default `out/L${N_LAYER}-D${N_EMBD}-${MODEL_TYPE}` is fine.         |

Then inside the `python train.py …` call, also update:

- `--data_file "data/<prefix>"` (no extension)
- `--vocab_size` (65536 for world tokenizer, 50304 for Pile/NeoX)
- `--my_exit_tokens <total tokens in your binidx>`
- `--magic_prime <value from step 4>`
- `--head_size 64` — keep at 64 for x070 (the CUDA kernel hard-codes 64).

Run:

```bash
cd RWKV-v7/train_temp
sh ./demo-training-prepare.sh
```

It will JIT-compile the CUDA kernels, build the model on CPU (`--accelerator cpu`), write `out/.../rwkv-init.pth`, and exit. Total time is a couple of minutes.

> Tip: Stage 1 also accepts `--load_model <path>` to *combine* weights from another checkpoint into the freshly-initialized model (with bilinear resizing if shapes differ). See `generate_init_weight` in `src/trainer.py:158`. Useful for upscaling.

---

## 6. Stage 3 — training run

`demo-training-run.sh` runs `train.py` with `--train_stage 3`. The trainer:

1. Scans `PROJ_DIR` for the highest-numbered `rwkv-<N>.pth` (or `rwkv-init.pth` if nothing else exists) and loads it.
2. Sets `epoch_begin = max_p + 1`.
3. Runs cosine-decay LR from `lr_init` → `lr_final` over `my_exit_tokens` tokens (with linear warmup over `warmup_steps`).
4. Saves a checkpoint every `epoch_save` "miniepochs" where **1 miniepoch = 40320 samples = 40320 × ctx_len tokens**.
5. Saves `rwkv-final.pth` and exits when `my_exit_tokens` is reached.

Key knobs at the top of the script:

| Variable     | Default  | Meaning                                                                                                 |
|--------------|----------|---------------------------------------------------------------------------------------------------------|
| `M_BSZ`      | 16       | micro batch size **per GPU**. Real bsz = `num_nodes × devices × micro_bsz`. Must give `40320 / real_bsz` integer. |
| `LR_INIT`    | 6e-4     | use 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048. Halve again for fine-tuning.             |
| `LR_FINAL`   | 6e-5     | cosine target (typically `LR_INIT / 10`).                                                               |
| `GRAD_CP`    | 1        | gradient checkpointing. 1 = save VRAM (slower), 0 = faster (more VRAM).                                 |
| `HEAD_CHUNK` | 0        | LM-head chunking. 0 = fast/much VRAM; 4096 = ~80% LM-head VRAM saving (slower); 65536 = ~70% saving.    |
| `KERNEL`     | `@rwkv3` | `""` = default v1 kernel, `@rwkv3` = newer kernel (~20% faster on H100 and some consumer GPUs).         |
| `EPOCH_SAVE` | 10       | save checkpoint every N miniepochs.                                                                     |

> **Important:** the first three `rm` lines at the top of `demo-training-run.sh` delete previous checkpoints (`rwkv-*0.pth`, `rwkv-71.pth`, `rwkv-final.pth`). They exist so a clean re-run doesn't accidentally resume from stale weights. **Comment them out** if you want to resume.

Run:

```bash
sh ./demo-training-run.sh
```

Expected first-epoch losses (must be within ±0.01 of these, else something is wrong):

```
0 4.875856 131.0863 ...
1 4.028621 56.1834  ...
2 3.801625 44.7739  ...
...
```

(Column 2 = mean loss, column 3 = perplexity = `exp(loss)`.)

---

## 7. Resuming / fine-tuning

### 7a. Resume after Ctrl-C or crash

Just re-run `demo-training-run.sh` **with the `rm` lines commented out**. The Stage-3 logic in `train.py:111-132` automatically:

- lists all `rwkv-*.pth` in `PROJ_DIR`,
- picks the highest miniepoch number (treating `init` as `-1`),
- sets `args.load_model` to it and `args.epoch_begin = max_p + 1`,
- defaults `warmup_steps` to 10 if you set it `<0`.

If loading the most recent checkpoint fails (e.g., file was truncated by an OOM kill), it falls back to the previous one (`my_pile_prev_p`).

### 7b. Fine-tuning from a published RWKV-7 checkpoint

1. Drop the checkpoint into `PROJ_DIR` and rename it `rwkv-init.pth` (so the loader picks it up as miniepoch −1). **Or** point `--load_model <path>` explicitly.
2. Use **very small** LR, e.g., `--lr_init 1e-5 --lr_final 1e-5`.
3. Keep the same `N_LAYER`, `N_EMBD`, `head_size`, `vocab_size` as the base. If shapes differ, use Stage 1 with `--load_model` to do a shape-adapting copy (bilinear interpolation along the first axis — see `trainer.py:170-192`).
4. Set `--my_exit_tokens` to your total fine-tune budget so the cosine decay completes when you want it to.
5. If the base has a different vocab structure than yours, retokenize your data with the matching tokenizer.

Use `--load_partial 1` if your checkpoint is missing some keys — they'll be filled with freshly-initialized weights.

### 7c. Stage 2 vs Stage 3

In code, `train_stage >= 2` triggers the auto-resume logic; `train_stage == 1` triggers init-weight generation. The provided scripts use `1` and `3`. `train_stage = 2` is equivalent to `3` for resume purposes; the original distinction was a historical pile-specific mode. Just stick with the scripts' values.

---

## 8. Important CLI flags (cheat sheet)

From `train.py:18-58`:

| Flag                 | Default      | Notes                                                                                          |
|----------------------|--------------|------------------------------------------------------------------------------------------------|
| `--load_model`       | `""`         | full path with `.pth`. In Stage 3 with `"0"`, the resume scanner picks the latest.             |
| `--wandb`            | `""`         | wandb project name; empty disables wandb.                                                      |
| `--proj_dir`         | `out`        | where checkpoints + `train_log.txt` go.                                                        |
| `--random_seed`      | `-1`         | `-1` = no global seed (preferred for multi-GPU sampling diversity).                            |
| `--data_file`        |              | prefix path (no extension) to `.bin/.idx`.                                                     |
| `--data_type`        | `utf-8`      | **must be `binidx`** for this trainer (asserted).                                              |
| `--vocab_size`       | `0`          | set explicitly (65536 for world, 50304 for NeoX/Pile).                                         |
| `--ctx_len`          | `1024`       | match the `magic_prime` you computed.                                                          |
| `--epoch_steps`      | overwritten  | trainer recomputes as `40320 // real_bsz`.                                                     |
| `--epoch_count`      | overwritten  | trainer recomputes as `magic_prime // 40320`.                                                  |
| `--epoch_begin`      | `0`          | overwritten in Stage 3 resume.                                                                 |
| `--epoch_save`       | `5`          | save every N miniepochs.                                                                       |
| `--micro_bsz`        | `12`         | per-GPU batch size.                                                                            |
| `--n_layer`          | `6`          | depth.                                                                                         |
| `--n_embd`           | `512`        | width.                                                                                         |
| `--dim_att`          | `0`          | `0` → defaults to `n_embd`.                                                                    |
| `--dim_ffn`          | `0`          | `0` → defaults to `int((n_embd*3.5)//32*32)`.                                                  |
| `--lr_init`          | `6e-4`       | see scaling table above.                                                                       |
| `--lr_final`         | `1e-5`       | cosine target.                                                                                 |
| `--warmup_steps`     | `-1`         | `-1` = no warmup; auto-set to 10 when resuming.                                                |
| `--beta1`/`--beta2`  | `0.9/0.99`   | Adam betas.                                                                                    |
| `--adam_eps`         | `1e-18`      | tiny eps is important for RWKV stability.                                                      |
| `--grad_cp`          | `0`          | 1 = gradient checkpointing (slower, less VRAM).                                                |
| `--weight_decay`     | `0`          | 0.001 for small, 0.1 for Pile-scale. **Only applied to params marked `wdecay` in `model.py`.** |
| `--grad_clip`        | `1.0`        | drop to 0.7/0.5/0.3/0.2 if you see loss spikes on bad samples.                                 |
| `--train_stage`      | `0`          | `1` = init-only, `3` = train (with auto-resume).                                               |
| `--ds_bucket_mb`     | `200`        | DeepSpeed bucket size. 2 for consumer GPUs, 200 for A100/H100. Buggy in newest DS — usually leave default. |
| `--head_size`        | `64`         | **keep 64 for x070** (CUDA kernels hard-code 64).                                              |
| `--head_chunk`       | `0`          | 0 = fast/big VRAM, 4096 = max saving, 65536 = middle.                                          |
| `--load_partial`     | `0`          | 1 = fill missing keys with fresh init.                                                         |
| `--magic_prime`      | `0`          | required.                                                                                      |
| `--my_testing`       | `x070`       | model variant tag (controls model.py code paths).                                              |
| `--kernel`           | `""`         | `@rwkv3` = newer/faster kernel.                                                                |
| `--my_exit_tokens`   | `0`          | total tokens; triggers cosine decay schedule + `rwkv-final.pth` save when reached.             |

Pytorch-Lightning Trainer flags (passed via `Trainer.add_argparse_args`):

- `--accelerator gpu|cpu`
- `--devices 1` (GPUs per node)
- `--num_nodes 1`
- `--precision bf16` (recommended; fp16 may overflow, fp32 is very slow)
- `--strategy deepspeed_stage_2` (Stage 3 also supported but `RWKV_JIT_ON` is disabled then)
- `--enable_progress_bar True`

---

## 9. Multi-GPU / multi-node

Set `--devices <gpus_per_node>` and `--num_nodes <N>`. `real_bsz = num_nodes × devices × micro_bsz` must divide `40320`. Convenient choices: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 28, 30, 32, 35, 40, 42, …

DeepSpeed Stage 2 is the default and works well up to 1.5 B params on commodity GPUs. For very large models, switch to Stage 3 (`--strategy deepspeed_stage_3`); note that this disables JIT and changes the checkpoint format (`my_save` uses `trainer.save_checkpoint(..., weights_only=True)` in that case).

For multi-node, launch with the usual `deepspeed --hostfile=… train.py …` or `torchrun --nnodes=… --nproc_per_node=…` wrappers; the script itself only needs `num_nodes` and `devices` set correctly.

---

## 10. VRAM tuning

Roughly from highest to lowest impact:

1. **`micro_bsz`** — linear in activation VRAM. Halve it first when OOM.
2. **`grad_cp=1`** — gradient checkpointing across blocks; ~30–50% activation savings, ~20% slower.
3. **`head_chunk`** — chunks the LM head loss computation. The LM head is `vocab × n_embd` and dominates for small models with large vocab.
   - `0` = no chunking, fastest, biggest VRAM.
   - `65536` = chunk size 64k, ~70% LM-head saving.
   - `4096` = chunk size 4k, ~80% saving, slowest.
4. **Kernel choice** — `--kernel @rwkv3` uses the H100-tuned clampw kernel; it's also faster on many consumer GPUs.
5. **DeepSpeed Stage 3** — partition optimizer + grads + params across GPUs; for very large models.
6. **Precision** — bf16 is the only sane choice for training. fp32 is supported but ~3× slower.

Note: `dim_att = n_embd` and `head_size = 64`, so the number of heads is `n_embd / 64`. The wkv7 CUDA kernel asserts `T % CHUNK_LEN == 0` where `CHUNK_LEN = 16`, so make sure `ctx_len % 16 == 0`.

---

## 11. Logs, checkpoints, wandb

- **`train_log.txt`** — one line per miniepoch end:
  `<epoch> <mean_loss> <perplexity> <lr> <timestamp> <current_epoch>`
- **Checkpoints** — `rwkv-<N>.pth` (every `epoch_save` miniepochs) and `rwkv-final.pth` (on completion).
- **`wandb`** — pass `--wandb <project>` to enable; logs `loss`, `lr`, `wd`, `Gtokens`, `kt/s` per step.

`epoch_count` is auto-set to `magic_prime // 40320`, which is the target number of miniepochs to walk the dataset once. Lightning will keep running past that (`max_epochs = -1`) but `rwkv-final.pth` is written and `exit(0)` is called as soon as `my_exit_tokens` worth of tokens has been seen.

---

## 12. Common pitfalls

- **`assert is_prime(args.magic_prime)` or ratio assert fires** → you changed `ctx_len` or `data_file` without recomputing `magic_prime`. Re-run `compute_magic_prime.py`.
- **`40320 % real_bsz != 0`** → choose a `micro_bsz` such that `num_nodes × devices × micro_bsz` divides 40320.
- **Process hangs at "Loading extension rwkv7_…"** → stale lock file in `~/.cache/torch_extensions/`. `rm -rf` that folder and retry.
- **First-epoch loss not within ±0.01 of the reference** → wrong vocab size, wrong tokenizer, wrong magic_prime, or wrong precision (don't use fp16). The reference numbers in `README.md` assume bf16 + MiniPile + the default config exactly.
- **Resume picks the wrong file** → check `PROJ_DIR` for stray `rwkv-*.pth`; the loader takes the highest integer in the filename.
- **OOM on the very first step** → reduce `micro_bsz`, set `grad_cp=1`, then `head_chunk=4096`. On consumer GPUs also try `--ds_bucket_mb 2`.
- **`pytorch_lightning` version mismatch** → must be exactly **1.9.5**. Other versions silently change Trainer API behavior.
- **Editing `head_size`** → don't, unless you also edit `HEAD_SIZE` and the hard-coded `64` inside the CUDA kernels. The model asserts `HEAD_SIZE == 64` at import.

---

## Quick recipe — your own data, from scratch

```bash
# 1. Build binidx (in v5 folder because that's where make_data.py lives)
cd /home/mechu/rwkv-lm/RWKV-v5
python make_data.py /path/to/mydata.jsonl 1 512
# Note the printed --my_exit_tokens N --magic_prime P --ctx_len 512 line.
mv mydata.bin mydata.idx /home/mechu/rwkv-lm/RWKV-v7/train_temp/data/

# 2. Edit the two shell scripts: set --data_file "data/mydata",
#    --my_exit_tokens N, --magic_prime P, --ctx_len 512, --vocab_size 65536
cd /home/mechu/rwkv-lm/RWKV-v7/train_temp

# 3. Init weights
sh ./demo-training-prepare.sh

# 4. Train (comment the 'rm' lines if resuming later)
sh ./demo-training-run.sh
```
