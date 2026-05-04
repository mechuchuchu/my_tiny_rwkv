# RWKV-7 KVW-Decay Training Guide

This folder contains an experimental variant of RWKV-7 that implements the ★★ KVW-Decay (Key-Value-Weight Decay) idea from [`rand_vec_ideas_en.md`](../../rand_vec_ideas_en.md). It includes code for comparing against baseline RWKV-7.

---

## 1. KVW-Decay at a Glance

**Baseline RWKV-7 decay (summary):**
```
w_logit = w0 + tanh(xw @ w1) @ w2          # xw = token-shifted input
w_decay = exp(-exp(w_logit))               # handled inside CUDA kernel, ∈ (0,1)
```

**KVW-Decay modification:**
```
w_logit_base = w0 + tanh(xw @ w1) @ w2     # same as before
wkv_concat   = cat([w_logit_base, k, v], -1)  # [B, T, 3C]
w_logit      = w_logit_base + (wkv_concat @ g_ww + b_ww)  # state-aware modulation
w_decay      = exp(-exp(w_logit))          # CUDA kernel unchanged
```

**Core intuition:** Lets the model ask itself: "Is what I'm writing into memory important enough to overwrite what's already there?" — by directly injecting k and v information into the decay computation.

**Implementation location:** `RWKV_Tmix_x070` class in `src/model.py`
- `__init__`: adds `self.g_ww` (3C, C) and `self.b_ww` (1, 1, C), **both zero-initialized**
- `forward`: KVW modulation applied right after value-residual gating

**Why zero-init matters:** At step 0, KVW-Decay is numerically identical to baseline RWKV-7. As training progresses, `g_ww` and `b_ww` converge to meaningful values.

**Parameter overhead:** `3C² + C` per layer (e.g. C=768 → ~1.77M/layer, L=12 → ~21M total). About +20% for a 0.1B model. To reduce, see the LoRA decomposition in the last section.

---

## 2. Environment Setup

```bash
# Requires Python 3.10+, CUDA 12.x, NVIDIA GPU
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==1.9.5 deepspeed ninja --upgrade
```

**Version notes:**
| Package | Recommended Version | Notes |
|---|---|---|
| torch | 2.5+ | Latest recommended |
| cuda toolkit | 12.5+ | System CUDA |
| pytorch-lightning | **must be 1.9.5** | Trainer API compatibility |
| deepspeed | latest | ZeRO stage 2/3 |
| ninja | latest | CUDA kernel JIT build |

`pytorch-lightning==1.9.5` is mandatory. 2.x removes `Trainer.from_argparse_args` and other APIs that will break things.

---

## 3. Hardware Requirements

Based on default config (`L12-D768`, ctx_len 512, micro_bsz 16):

| GPU VRAM | Recommended Settings |
|---|---|
| 10~12GB | `--micro_bsz 4 --grad_cp 1` |
| 16~24GB | `--micro_bsz 8~16 --grad_cp 1` |
| 40GB+ | `--micro_bsz 32+ --grad_cp 0` |
| 80GB | larger bsz, grad_cp 0 |

KVW-Decay uses slightly more VRAM than baseline (due to the 3C-wide concat tensor). If tight, reduce `micro_bsz` by 1–2.

`grad_cp=1` enables gradient checkpointing (~20% speed penalty, significant VRAM savings).

---

## 4. Quickstart — MiniPile (1.5B tokens)

For validation. Use this to confirm things work on a single GPU.

### 4.1 Download Data

```bash
cd RWKV-v7-kvw-decay/train_temp
mkdir -p data
wget --continue -O data/minipile.idx \
  https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget --continue -O data/minipile.bin \
  https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

### 4.2 Generate Initial Weights

```bash
sh ./demo-training-prepare.sh
```

This produces `out/L12-D768-x070/rwkv-init.pth`.

### 4.3 Run Training

```bash
sh ./demo-training-run.sh
```

On success, `out/L12-D768-x070/train_log.txt` will contain logs like this (since KVW-Decay is identical to baseline at step 0, you can directly compare loss curves):

```
0 4.875856 131.0863 0.00059975 ...
1 4.028621 56.1834  0.00059899 ...
...
```

---

## 5. Training on Custom Datasets

### 5.1 Data Format — `binidx`

This trainer **only accepts binidx format** (`assert args.data_type in ["binidx"]`).
- `.bin`: raw byte sequence of token IDs (uint16 or uint32)
- `.idx`: index of position/length for each document

Prepare raw text in **JSONL** format (one JSON per line, using the `text` field):

```jsonl
{"text": "First document body goes here."}
{"text": "Second document. Can include\nnewlines."}
{"text": "Mixed Korean and English works fine."}
```

> **Note:** RWKV is typically trained on raw text, so only the `text` field is needed. For chat or instruction formats, pack them directly into the `text` field (`User: ...\n\nAssistant: ...` etc.).

### 5.2 Tokenizer Selection

| Tokenizer | vocab_size | Use case |
|---|---|---|
| `rwkv_vocab_v20230424` (TRIE) | 65536 | Multilingual RWKV standard (Korean OK) |
| Pile / GPT-NeoX BPE | 50304 | English-focused |

The RWKV TRIE tokenizer is recommended by default. It has good byte efficiency for Korean/Japanese/Chinese.

### 5.3 JSONL → binidx Conversion

`make_data.py` lives in the RWKV-v5 folder and can be used as-is:

```bash
# 1) Copy tokenizer and conversion script
cp ../../RWKV-v5/make_data.py .
cp -r ../../RWKV-v5/tokenizer .

# 2) Put your data inside data/
mkdir -p data
cp /path/to/your_corpus.jsonl data/

# 3) Convert: jsonl, n_epoch (shuffle repetitions), ctx_len
python make_data.py data/your_corpus.jsonl 3 4096
# => produces data/your_corpus.bin / data/your_corpus.idx
```

Copy the last line printed by the script directly into your training command:
```
--my_exit_tokens 12345678 --magic_prime 24107 --ctx_len 4096
```

> `n_epoch` = how many times to shuffle and duplicate the data. Use 3–5 for fine-tuning / small data, 1 for large-scale pretraining.

### 5.4 What is `magic_prime`? — Important

When slicing a dataset into ctx_len chunks, RWKV uses a **prime mod hash** to determine sample positions (`__getitem__` in `src/dataset.py`):

```
magic_prime = largest prime of the form (3n+2) that is less than (datalen / ctx_len - 1)
```

Conditions:
- `is_prime(magic_prime)` — must be prime
- `magic_prime % 3 == 2` — must be of the form `3n+2`
- `0.9 < magic_prime / (datalen // ctx_len) <= 1.0`

`make_data.py` computes this automatically. If you need to compute it manually for an existing binidx:
- Count down from `(token count / ctx_len - 1)` until you hit a value that's prime and `%3==2`
- Or use https://www.dcode.fr/prime-numbers-search

> **Changing `ctx_len` requires recomputing `magic_prime`.** Otherwise `MyDataset.__init__` will throw an assertion error.

---

## 6. Building Training Commands (Script Guide)

### 6.1 Prepare Stage (`demo-training-prepare.sh`)

`train_stage 1` — creates initial weights and exits after 1 epoch. You can kill it immediately (runs in CPU mode, fast). Key arguments:

```bash
python train.py \
  --proj_dir out/L12-D768-x070 \
  --data_file "data/your_corpus" \   # prefix only, no .bin/.idx extension
  --data_type "binidx" \
  --vocab_size 65536 \                # match your tokenizer
  --my_testing x070 \                 # RWKV-7 mode (KVW-Decay is built on top of x070)
  --ctx_len 512 \
  --train_stage 1 \
  --n_layer 12 --n_embd 768 --head_size 64 \
  --my_exit_tokens 1498226207 --magic_prime 2926181 \
  --accelerator cpu --devices 1 --precision bf16 \
  --strategy deepspeed_stage_2 --grad_cp 1
```

### 6.2 Run Stage (`demo-training-run.sh`)

`train_stage 3` — actual training. Automatically resumes from the latest checkpoint if one exists.

```bash
python train.py \
  --load_model "0" \                  # auto-detects last checkpoint
  --proj_dir out/L12-D768-x070 \
  --my_testing x070 \
  --ctx_len 512 --train_stage 3 \
  --epoch_count 999999 --epoch_begin 0 \
  --data_file "data/your_corpus" \
  --my_exit_tokens 1498226207 --magic_prime 2926181 \
  --num_nodes 1 --micro_bsz 16 \
  --n_layer 12 --n_embd 768 --head_size 64 \
  --lr_init 6e-4 --lr_final 6e-5 \
  --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
  --weight_decay 0.001 --epoch_save 10 \
  --vocab_size 65536 --data_type "binidx" \
  --accelerator gpu --devices 1 --precision bf16 \
  --strategy deepspeed_stage_2 --grad_cp 1 \
  --enable_progress_bar True
```

> **`--load_model "0"` is a placeholder.** When `train_stage >= 2`, the trainer automatically loads the latest `rwkv-*.pth` from `proj_dir` (`train.py:107`).

### 6.3 Argument Cheat Sheet

| Argument | Meaning | Recommended |
|---|---|---|
| `--n_layer` | Number of blocks | 0.1B≈12, 0.4B≈24, 1.5B≈24 |
| `--n_embd` | Embedding dimension (must be multiple of 32 and divisible by head_size) | 768/1024/2048 |
| `--head_size` | Head dimension | **fixed at 64** (assumed by CUDA kernel) |
| `--ctx_len` | Context length | 512/1024/4096; must be multiple of 16 |
| `--micro_bsz` | Batch size per GPU | depends on VRAM |
| `--lr_init` / `--lr_final` | Cosine warmup-decay endpoints | L12-D768: 6e-4→6e-5, L24-D2048: 3e-4→3e-5 |
| `--beta2` | Adam β2 | 0.99 (small-scale), 0.999 (large-scale long-run) |
| `--adam_eps` | Adam ε | 1e-8 (init), 1e-18 (run) |
| `--weight_decay` | wd. **Applied only to large tensors with `.weight` in name** | 0.001~0.1 |
| `--grad_clip` | Gradient clipping | 1.0 default; try 0.5/0.3 if unstable |
| `--grad_cp` | Gradient checkpointing | 1 = save VRAM, 0 = faster |
| `--precision` | Float mode | **`bf16` strongly recommended** (fp16 overflows, fp32 very slow) |
| `--strategy` | DeepSpeed stage | `deepspeed_stage_2` default; stage_3 for 70B+ (disables JIT) |
| `--epoch_save` | Save every N miniepochs | 10~50 (1 miniepoch = 40320 × ctx_len tokens) |
| `--my_exit_tokens` | Total tokens in dataset | from `make_data.py` output |
| `--magic_prime` | See §5.4 | depends on data + ctx_len |

> **`epoch_count` is computed automatically:** `train.py:103` overrides it as `args.epoch_count = args.magic_prime // 40320`. Whatever you pass on CLI (e.g. 999999) is ignored; it's set to cover exactly one pass through the data.

> **1 miniepoch = 40320 samples** (fixed). Since `real_bsz × steps = 40320`, you need `40320 % real_bsz == 0` (i.e. `num_nodes × devices × micro_bsz` must be a divisor of 40320).

---

## 7. Multi-GPU / Multi-Node

```bash
# 1 node, 8 GPUs
python train.py ... --num_nodes 1 --devices 8 --micro_bsz 8 \
  --strategy deepspeed_stage_2

# Multi-node (e.g. 2 nodes × 8 GPUs)
# Run the same command with --num_nodes 2 --devices 8 on each node.
# Requires inter-node communication setup via DeepSpeed launcher or torchrun.
```

`real_bsz = num_nodes × devices × micro_bsz`. Adjust `micro_bsz` so this is a divisor of 40320.

ZeRO stage comparison:
- **stage 2**: optimizer state + gradient sharding. Recommended. JIT kernels work.
- **stage 3**: parameter sharding. For very large models only. RWKV trainer auto-sets `RWKV_JIT_ON=0` for stage 3 (`train.py:177`).

---

## 8. Resuming / Fine-tuning / Weight Conversion

### Resume (auto)
With `train_stage 3`, the trainer auto-loads the highest-numbered `rwkv-{N}.pth` from `proj_dir`. To restart from scratch, the `rm` line in `demo-training-run.sh` handles that.

### Restart from scratch
```bash
rm out/L12-D768-x070/rwkv-*.pth
sh demo-training-prepare.sh
sh demo-training-run.sh
```

### Fine-tuning from another model (e.g. baseline RWKV-7 → KVW-Decay)
```bash
# Copy baseline weights as rwkv-init.pth
cp /path/to/baseline-rwkv7.pth out/L12-D768-x070/rwkv-init.pth

# KVW-Decay's new parameters (g_ww, b_ww) are missing, so use partial load
python train.py ... --train_stage 3 --load_partial 1 \
  --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10
```

`--load_partial 1` uses model init values (i.e. zeros) for any keys missing from the checkpoint (`train.py:231`). Since KVW-Decay parameters are zero-initialized, **the model at this point is numerically identical to baseline**, and fine-tuning will train the KVW parameters from there.

### Ablation: freeze KVW only (g_ww, b_ww)
Add `.requires_grad = False` to `self.g_ww` and `self.b_ww` in `model.py` to get exactly the same dynamics as baseline.

---

## 9. Monitoring Training

### 9.1 Text Logs
- `out/L12-D768-x070/train_log.txt` — one line per miniepoch: `epoch loss ppl lr timestamp kt_s`
- If normal, loss should be **within ±0.01** of the baseline log near step 0 (guaranteed by zero-init).
- With `--enable_progress_bar True`, lr / loss / Kt/s are shown in real time on stdout.

### 9.2 Checkpoints
- `rwkv-init.pth` — initial weights
- `rwkv-{epoch}.pth` — saved every `epoch_save` miniepochs
- `rwkv-final.pth` — saved on full training completion

### 9.3 KVW-Decay Debug Tips
Check that `g_ww` and `b_ww` are actually changing during training:
```python
# In inference or a separate script
import torch
ckpt = torch.load("rwkv-N.pth", map_location="cpu")
for k, v in ckpt.items():
    if "g_ww" in k or "b_ww" in k:
        print(f"{k}: mean={v.mean():.4e}, std={v.std():.4e}, abs_max={v.abs().max():.4e}")
```

If `std` is moving away from 0, training is working. If all layers stay near 0, the learning signal is too weak → try a slightly higher lr, or use a small random init for `g_ww`.

---

## 10. Common Issues

| Symptom | Cause / Fix |
|---|---|
| `assert is_prime(args.magic_prime)` fails | Recompute per §5.4 |
| `assert args.epoch_steps * args.real_bsz == 40320` fails | Adjust so `num_nodes × devices × micro_bsz` divides 40320 |
| OOM | Reduce `micro_bsz` → enable `grad_cp 1` → reduce `ctx_len` |
| Loss is NaN/inf | Switch `fp16`→`bf16`, set `grad_clip 0.5`, lower lr |
| Loss differs from baseline at step 0 | `g_ww` or `b_ww` may not be zero-initialized. Check `generate_init_weight` handling |
| CUDA kernel JIT build fails | Install `ninja` / `cuda-toolkit` / verify `nvcc --version` works |
| `pytorch_lightning` import error | Must use `pytorch-lightning==1.9.5` |
| Multi-node error on single GPU | Explicitly set `--num_nodes 1 --devices 1` |
| `RuntimeError: ...JIT...` (stage 3) | Expected. Stage 3 auto-disables JIT (`train.py:177`) |

---

## 11. Further Experiment Ideas for KVW-Decay

### 11.1 Ablation
- Freeze `g_ww`, train only `b_ww` → measure the effect of channel-wise bias alone
- `wkv_concat = cat([w, k, v])` → `cat([w, k])` (drop v) → measure v's contribution
- `wkv_concat = cat([k, v])` (drop base w) → replace w_logit entirely

### 11.2 LoRA Decomposition (saves params, 21M → 2.4M)
Modify in `model.py`:

```python
# Original
self.g_ww = nn.Parameter(torch.zeros(3*C, C))
self.b_ww = nn.Parameter(torch.zeros(1, 1, C))
# forward: w = w + (wkv_concat @ self.g_ww + self.b_ww)

# LoRA form
D_KVW_LORA = max(32, int(round((2.5*(C**0.5))/32)*32))
self.gw1 = nn.Parameter(torch.zeros(3*C, D_KVW_LORA))         # zero-init
self.gw2 = nn.Parameter(ortho_init(torch.zeros(D_KVW_LORA, C), 0.1))  # ortho-init
self.b_ww = nn.Parameter(torch.zeros(1, 1, C))
# forward: w = w + ((wkv_concat @ self.gw1) @ self.gw2 + self.b_ww)
# gw1=0 guarantees identity to baseline at initialization
```

This form is consistent with other LoRA pairs in RWKV-7 (`w1/w2`, `a1/a2`, `v1/v2`, `g1/g2`).

### 11.3 Sigmoid Form (original proposal)
To use `sigmoid(wkv_concat @ g_ww + b_ww)` as in the original idea, a logit-space transform is needed. Either bypass the CUDA kernel or apply this mapping:
```
w_decay = sigmoid(z)              # (0,1)
w_logit = log(-log(w_decay)) = log(log(1+exp(-z)))  # log of softplus(-z)
```
This is numerically unstable, so the current implementation uses additive logit-space form instead. Recommended only as a separate ablation variant.

### 11.4 Baseline Comparison Protocol
For a fair comparison:
1. Train baseline with the same data/hyperparameters in `RWKV-v7/train_temp/`
2. Train KVW-Decay with identical settings in `RWKV-v7-kvw-decay/train_temp/`
3. Compare loss/ppl at the same token count
4. Also run downstream benchmarks (`rwkv_mmlu_eval.py` etc.) at the same token count

The key question is whether KVW-Decay's extra parameters (~+20%) yield meaningful improvement — for an honest comparison, also run a parameter-matched baseline (slightly larger n_embd to equalize parameter count).

---

## 12. Reference Files

| File | Purpose |
|---|---|
| `train.py` | Training entry point (argparse + Lightning Trainer) |
| `src/model.py` | **KVW-Decay modification location** (`RWKV_Tmix_x070`) |
| `src/dataset.py` | binidx dataset, magic_prime validation |
| `src/trainer.py` | LR schedule, logging/checkpoint callbacks |
| `src/binidx.py` | Megatron-style binidx reader |
| `cuda/*` | RWKV-7 fused CUDA kernels (no modification needed) |
| `rwkv7_train_simplified.py` | (Reference only) simplified single-file training demo — separate from this folder |
| `../../RWKV-v5/make_data.py` | JSONL → binidx converter |
| `../../RWKV-v5/tokenizer/` | RWKV TRIE tokenizer |

---

## 13. Weight Table (Reference)

Only parameters added/modified by KVW-Decay (based on 1.5B, L24-D2048):

| name | shape | comment | initialization |
|---|---|---|---|
| `blocks.*.att.g_ww` | [3·2048, 2048] = [6144, 2048] | KVW-Decay (new) | **0** |
| `blocks.*.att.b_ww` | [1, 1, 2048] | KVW-Decay (new) | **0** |

All other parameters are identical to baseline RWKV-7. Since `g_ww`/`b_ww` do not have `.weight` in their names:
- `configure_optimizers`: placed in lr_1x group (no weight decay)
- `generate_init_weight`: flows into the first branch, preserving zeros

This means weight decay is intentionally not applied to the new parameters. If you decide wd is needed, explicitly add `g_ww` to `lr_decay` in `configure_optimizers` in `model.py`.
