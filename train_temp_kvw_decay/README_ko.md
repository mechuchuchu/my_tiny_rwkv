# RWKV-7 KVW-Decay 학습 가이드

이 폴더는 [`rand_vec_ideas_en.md`](../../rand_vec_ideas_en.md)의 ★★ KVW-Decay (Key-Value-Weight Decay) 아이디어를 RWKV-7 위에 얹은 변형판입니다. baseline RWKV-7과 비교하기 위한 실험 코드가 담겨 있습니다.

---

## 1. KVW-Decay가 뭔지 한눈에

**기존 RWKV-7의 decay (요약):**
```
w_logit = w0 + tanh(xw @ w1) @ w2          # xw = token-shifted input
w_decay = exp(-exp(w_logit))               # CUDA kernel 내부에서 처리, ∈ (0,1)
```

**KVW-Decay 수정:**
```
w_logit_base = w0 + tanh(xw @ w1) @ w2     # 기존과 동일
wkv_concat   = cat([w_logit_base, k, v], -1)  # [B, T, 3C]
w_logit      = w_logit_base + (wkv_concat @ g_ww + b_ww)  # state-aware modulation
w_decay      = exp(-exp(w_logit))          # CUDA kernel 그대로 사용
```

**핵심 직관:** "지금 메모리에 쓰고 있는 게 기존 내용을 덮어쓸 만큼 중요한가?"라는 질문을 모델이 자체적으로 던질 수 있게 한다 (k, v 정보를 decay 계산에 직접 주입).

**구현 위치:** `src/model.py`의 `RWKV_Tmix_x070` 클래스
- `__init__`: `self.g_ww` (3C, C), `self.b_ww` (1, 1, C) 추가, **둘 다 zero-init**
- `forward`: value-residual gating 직후 KVW 변조 적용

**Zero-init 의의:** 학습 step 0에서 baseline RWKV-7과 **수치상 완전히 동일**. 학습이 진행되면서 `g_ww`, `b_ww`가 의미 있는 값으로 수렴.

**파라미터 오버헤드:** 레이어당 `3C² + C` (예: C=768이면 ~1.77M/layer, L=12면 ~21M 추가). 0.1B 모델 기준 약 +20%. 더 줄이려면 LoRA 분해 (마지막 섹션 참고).

---

## 2. 환경 설정

```bash
# Python 3.10+, CUDA 12.x, NVIDIA GPU 필요
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==1.9.5 deepspeed ninja --upgrade
```

**버전 주의사항:**
| 패키지 | 권장 버전 | 비고 |
|---|---|---|
| torch | 2.5+ | 최신 권장 |
| cuda toolkit | 12.5+ | 시스템 cuda |
| pytorch-lightning | **반드시 1.9.5** | trainer API 호환성 |
| deepspeed | 최신 | ZeRO stage 2/3 |
| ninja | 최신 | CUDA kernel JIT 빌드 |

`pytorch-lightning==1.9.5`는 강제입니다. 2.x는 `Trainer.from_argparse_args` 등이 제거되어 동작하지 않습니다.

---

## 3. 하드웨어 요구사항

기본 설정 (`L12-D768`, ctx_len 512, micro_bsz 16) 기준:

| GPU VRAM | 권장 설정 |
|---|---|
| 10~12GB | `--micro_bsz 4 --grad_cp 1` |
| 16~24GB | `--micro_bsz 8~16 --grad_cp 1` |
| 40GB+ | `--micro_bsz 32+ --grad_cp 0` |
| 80GB | bsz 더 크게, grad_cp 0 |

KVW-Decay는 baseline 대비 약간 더 많은 VRAM을 씁니다 (3C-wide concat 텐서). 빠듯하면 `micro_bsz`를 1~2 줄이세요.

`grad_cp=1`은 gradient checkpointing 활성화 (속도 ~20% 손해, VRAM 큰폭 절약).

---

## 4. 빠른 시작 — MiniPile (1.5B tokens)

검증용. 1 GPU에서 동작 확인할 때 사용.

### 4.1 데이터 다운로드

```bash
cd RWKV-v7-kvw-decay/train_temp
mkdir -p data
wget --continue -O data/minipile.idx \
  https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget --continue -O data/minipile.bin \
  https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

### 4.2 초기 가중치 생성

```bash
sh ./demo-training-prepare.sh
```

`out/L12-D768-x070/rwkv-init.pth`가 생성됩니다.

### 4.3 학습 실행

```bash
sh ./demo-training-run.sh
```

성공 시 `out/L12-D768-x070/train_log.txt`에 다음과 비슷한 로그 (KVW-Decay는 step 0에서 baseline과 동일하므로 baseline 로스 곡선과 비교 가능):

```
0 4.875856 131.0863 0.00059975 ...
1 4.028621 56.1834  0.00059899 ...
...
```

---

## 5. 커스텀 데이터셋으로 학습

### 5.1 데이터 형식 — `binidx`

이 trainer는 **binidx 형식만** 받습니다 (`assert args.data_type in ["binidx"]`).
- `.bin`: 토큰 ID들의 raw byte 시퀀스 (uint16 또는 uint32)
- `.idx`: 각 document의 위치/길이 인덱스

원본 텍스트는 **JSONL** 형식으로 준비합니다 (한 줄에 하나의 JSON, `text` 필드 사용):

```jsonl
{"text": "여기에 첫 번째 문서의 본문이 들어갑니다."}
{"text": "두 번째 문서. 줄바꿈\n포함 가능."}
{"text": "Mixed Korean and English works fine."}
```

> **주의:** RWKV는 raw text 학습이 일반적이라 `text` 필드만 쓰면 됩니다. 채팅 포맷이나 instruction 포맷이 필요하면 그 형태대로 `text` 안에 직접 합쳐 넣으세요 (`User: ...\n\nAssistant: ...` 등).

### 5.2 토크나이저 선택

| 토크나이저 | vocab_size | 용도 |
|---|---|---|
| `rwkv_vocab_v20230424` (TRIE) | 65536 | 다국어 RWKV 표준 (한국어 OK) |
| Pile / GPT-NeoX BPE | 50304 | 영어 위주 |

기본은 RWKV TRIE 토크나이저를 추천. 한국어/일본어/중국어 byte 효율이 좋습니다.

### 5.3 JSONL → binidx 변환

`make_data.py`는 RWKV-v5 폴더에 있어 그대로 가져다 씁니다:

```bash
# 1) 토크나이저와 변환 스크립트 가져오기
cp ../../RWKV-v5/make_data.py .
cp -r ../../RWKV-v5/tokenizer .

# 2) 데이터를 data/ 안에 둠
mkdir -p data
cp /path/to/your_corpus.jsonl data/

# 3) 변환: jsonl, n_epoch (셔플 반복 횟수), ctx_len
python make_data.py data/your_corpus.jsonl 3 4096
# => data/your_corpus.bin / data/your_corpus.idx 생성
```

스크립트 마지막에 출력되는 라인을 그대로 학습 명령에 복붙합니다:
```
--my_exit_tokens 12345678 --magic_prime 24107 --ctx_len 4096
```

> `n_epoch`는 셔플하면서 데이터를 몇 번 복제할지. 파인튜닝/소규모 데이터면 3~5, pretrain용 대규모면 1.

### 5.4 `magic_prime`이 뭔가요? — 꼭 이해하세요

데이터셋을 ctx_len 단위로 끊어 인덱싱할 때 RWKV는 **prime mod hash**로 샘플 위치를 결정합니다 (`src/dataset.py`의 `__getitem__`). 이때:

```
magic_prime = (datalen / ctx_len - 1)보다 작은 가장 큰 (3n+2)형 prime
```

조건:
- `is_prime(magic_prime)` — 소수
- `magic_prime % 3 == 2` — `3n+2` 형태
- `0.9 < magic_prime / (datalen // ctx_len) <= 1.0`

`make_data.py`가 자동 계산해 주지만, 이미 가지고 있는 binidx에 대해 직접 구해야 한다면:
- 토큰 수 / ctx_len - 1 부터 거꾸로 내려가며 `is_prime && %3==2` 첫 hit
- 또는 https://www.dcode.fr/prime-numbers-search 사용

> **`ctx_len`을 바꾸면 `magic_prime`도 반드시 다시 계산.** 안 그러면 `MyDataset.__init__`에서 assert 실패.

---

## 6. 학습 명령 만들기 (스크립트 가이드)

### 6.1 prepare 단계 (`demo-training-prepare.sh`)

`train_stage 1` — 초기 가중치만 만들고 1 epoch만 돌고 종료. 즉시 종료시켜도 OK (CPU 모드라 빠릅니다). 핵심 인자:

```bash
python train.py \
  --proj_dir out/L12-D768-x070 \
  --data_file "data/your_corpus" \   # .bin/.idx 확장자 빼고 prefix만
  --data_type "binidx" \
  --vocab_size 65536 \                # 토크나이저에 맞게
  --my_testing x070 \                 # RWKV-7 모드 (KVW-Decay도 x070 위에 얹힌 형태)
  --ctx_len 512 \
  --train_stage 1 \
  --n_layer 12 --n_embd 768 --head_size 64 \
  --my_exit_tokens 1498226207 --magic_prime 2926181 \
  --accelerator cpu --devices 1 --precision bf16 \
  --strategy deepspeed_stage_2 --grad_cp 1
```

### 6.2 run 단계 (`demo-training-run.sh`)

`train_stage 3` — 본 학습. 이전 체크포인트가 있으면 자동으로 이어 학습.

```bash
python train.py \
  --load_model "0" \                  # 자동 last checkpoint 검색
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

> **`--load_model "0"`은 placeholder.** 실제로는 `train_stage>=2`이면 `proj_dir`에서 가장 최근 `rwkv-*.pth`를 자동 로드 (`train.py:107`).

### 6.3 인자 cheat sheet

| 인자 | 의미 | 권장값 |
|---|---|---|
| `--n_layer` | block 개수 | 0.1B≈12, 0.4B≈24, 1.5B≈24 |
| `--n_embd` | 임베딩 차원 (32의 배수, head_size로 나눠떨어져야) | 768/1024/2048 |
| `--head_size` | head 차원 | **64 고정** (CUDA kernel 가정) |
| `--ctx_len` | 컨텍스트 길이 | 512/1024/4096; 16의 배수 |
| `--micro_bsz` | GPU 1개당 batch | VRAM 따라 |
| `--lr_init` / `--lr_final` | cosine warmup-decay 끝점 | L12-D768: 6e-4→6e-5, L24-D2048: 3e-4→3e-5 |
| `--beta2` | Adam β2 | 0.99 (소규모), 0.999 (대규모 long-run) |
| `--adam_eps` | Adam ε | 1e-8 (init), 1e-18 (run) |
| `--weight_decay` | wd. **`.weight` 들어간 큰 텐서에만 적용됨** | 0.001~0.1 |
| `--grad_clip` | gradient clipping | 1.0 기본, 불안정하면 0.5/0.3 |
| `--grad_cp` | gradient checkpointing | 1 = VRAM 절약, 0 = 빠름 |
| `--precision` | float 모드 | **`bf16` 강력 권장** (fp16는 overflow, fp32는 매우 느림) |
| `--strategy` | DeepSpeed stage | `deepspeed_stage_2` 기본; 70B+은 stage_3 (단 JIT 비활성화됨) |
| `--epoch_save` | N "miniepoch"마다 저장 | 10~50 (1 miniepoch = 40320 × ctx_len 토큰) |
| `--my_exit_tokens` | 데이터 총 토큰 수 | `make_data.py` 출력값 |
| `--magic_prime` | 위 §5.4 참고 | 데이터 + ctx_len에 종속 |

> **`epoch_count`는 자동 계산:** `train.py:103`에서 `args.epoch_count = args.magic_prime // 40320`로 덮어씁니다. CLI에 999999을 넣어도 무시되고 데이터 1회 통과 분량으로 정해집니다.

> **1 miniepoch = 40320 samples** (고정). real_bsz × steps = 40320 이어야 하므로 `40320 % real_bsz == 0`을 만족해야 합니다 (즉 `num_nodes × devices × micro_bsz`가 40320의 약수).

---

## 7. 멀티 GPU / 멀티 노드

```bash
# 1 노드 8 GPU
python train.py ... --num_nodes 1 --devices 8 --micro_bsz 8 \
  --strategy deepspeed_stage_2

# 멀티 노드 (예: 2 노드 × 8 GPU)
# 각 노드에서 동일 명령에 --num_nodes 2 --devices 8.
# DeepSpeed launcher 또는 torchrun으로 노드 간 통신 설정 필요.
```

`real_bsz = num_nodes × devices × micro_bsz`. 이 값이 40320의 약수가 되도록 micro_bsz 조정.

ZeRO stage 비교:
- **stage 2**: optimizer state + gradient sharding. 추천. JIT kernel 사용 가능.
- **stage 3**: parameter sharding 추가. 거대 모델 전용. RWKV trainer는 stage 3일 때 자동으로 `RWKV_JIT_ON=0` 설정 (`train.py:177`).

---

## 8. 학습 재개 / 파인튜닝 / 가중치 변환

### 재개 (auto)
`train_stage 3`로 돌리면 `proj_dir/rwkv-{N}.pth` 중 가장 큰 N을 자동 로드. 처음부터 다시 시작하려면 `demo-training-run.sh` 안의 `rm` 라인이 처리합니다.

### 처음부터 다시
```bash
rm out/L12-D768-x070/rwkv-*.pth
sh demo-training-prepare.sh
sh demo-training-run.sh
```

### 다른 모델로 파인튜닝 (e.g. baseline RWKV-7 → KVW-Decay)
```bash
# baseline 가중치를 rwkv-init.pth로 복사
cp /path/to/baseline-rwkv7.pth out/L12-D768-x070/rwkv-init.pth

# KVW-Decay 신규 파라미터(g_ww, b_ww)는 누락이므로 partial load 사용
python train.py ... --train_stage 3 --load_partial 1 \
  --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10
```

`--load_partial 1`은 체크포인트에 없는 키는 모델 init값(즉 zero)을 그대로 사용 (`train.py:231`). KVW-Decay 파라미터가 zero-init이므로 **이 시점에서 모델은 baseline과 수치상 동일**, 그 위에 fine-tune하면 KVW 파라미터가 학습됩니다.

### KVW만 ablation (g_ww, b_ww freeze)
원하면 `model.py`에서 `self.g_ww`, `self.b_ww`에 `.requires_grad = False` 추가하면 baseline과 정확히 같은 동역학.

---

## 9. 학습 모니터링

### 9.1 텍스트 로그
- `out/L12-D768-x070/train_log.txt` — 매 miniepoch마다 한 줄: `epoch loss ppl lr timestamp kt_s`
- 정상이면 baseline 로그(README 상단 표)와 **±0.01 이내** 차이여야 함 (zero-init 덕분에 step 0 가까이는 거의 동일).
- 진행 상황은 `--enable_progress_bar True` 로 stdout에 lr / loss / Kt/s 가 실시간 표시됨.

### 9.2 체크포인트
- `rwkv-init.pth` — 초기 가중치
- `rwkv-{epoch}.pth` — `epoch_save`마다
- `rwkv-final.pth` — 학습 완전 종료 시

### 9.3 KVW-Decay 별도 디버깅 팁
학습 중 `g_ww`, `b_ww`가 실제로 의미있게 움직이는지 확인:
```python
# inference 또는 별도 스크립트에서
import torch
ckpt = torch.load("rwkv-N.pth", map_location="cpu")
for k, v in ckpt.items():
    if "g_ww" in k or "b_ww" in k:
        print(f"{k}: mean={v.mean():.4e}, std={v.std():.4e}, abs_max={v.abs().max():.4e}")
```

`std`가 0에서 멀어지면 학습 중. 모든 레이어가 0 근처에 머무르면 학습 신호가 너무 약하다는 뜻 → lr 살짝 올리거나 `g_ww` 작은 random init 시도.

---

## 10. 자주 만나는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `assert is_prime(args.magic_prime)` 실패 | §5.4 다시 계산 |
| `assert args.epoch_steps * args.real_bsz == 40320` 실패 | `num_nodes × devices × micro_bsz`가 40320의 약수가 되도록 조정 |
| OOM | `micro_bsz` 줄이기 → `grad_cp 1` → `ctx_len` 줄이기 |
| Loss가 NaN/inf | precision을 `fp16`→`bf16`으로, `grad_clip 0.5`로, lr 낮추기 |
| Loss가 baseline 대비 step 0부터 다름 | `g_ww` 또는 `b_ww`가 zero-init이 아닐 가능성. `generate_init_weight` 처리 확인 |
| CUDA kernel JIT 빌드 실패 | `ninja` 설치 / `cuda-toolkit` 설치 / `nvcc --version` 동작 확인 |
| `pytorch_lightning` import 에러 | 반드시 `pytorch-lightning==1.9.5` |
| 1 GPU인데 multi-node 에러 | `--num_nodes 1 --devices 1` 명시 |
| `RuntimeError: ...JIT...` (stage 3) | 정상. stage 3는 자동으로 JIT 비활성화 (`train.py:177`) |

---

## 11. KVW-Decay 추가 실험 아이디어

### 11.1 Ablation
- `g_ww`만 freeze, `b_ww`만 학습 → channel-wise bias만의 효과
- `wkv_concat = cat([w, k, v])` → `cat([w, k])` (v 제외) → v의 기여도 측정
- `wkv_concat = cat([k, v])` (base w 제외) → 기존 w_logit 대체로 변경

### 11.2 LoRA 분해 (파라미터 절약, 21M → 2.4M)
`model.py`에서 다음과 같이 바꿉니다:

```python
# 기존
self.g_ww = nn.Parameter(torch.zeros(3*C, C))
self.b_ww = nn.Parameter(torch.zeros(1, 1, C))
# forward: w = w + (wkv_concat @ self.g_ww + self.b_ww)

# LoRA 형태
D_KVW_LORA = max(32, int(round((2.5*(C**0.5))/32)*32))
self.gw1 = nn.Parameter(torch.zeros(3*C, D_KVW_LORA))         # zero-init
self.gw2 = nn.Parameter(ortho_init(torch.zeros(D_KVW_LORA, C), 0.1))  # ortho-init
self.b_ww = nn.Parameter(torch.zeros(1, 1, C))
# forward: w = w + ((wkv_concat @ self.gw1) @ self.gw2 + self.b_ww)
# gw1=0이면 시작 시점에 baseline과 동일 보장
```

이 형태는 RWKV-7의 다른 LoRA들(`w1/w2`, `a1/a2`, `v1/v2`, `g1/g2`)과 일관됨.

### 11.3 Sigmoid 형태 (문서 원안)
원 idea의 `sigmoid(wkv_concat @ g_ww + b_ww)` 그대로 쓰려면 logit-space 변환이 필요. CUDA kernel을 우회하거나 다음과 같은 매핑을 거쳐야 함:
```
w_decay = sigmoid(z)              # (0,1)
w_logit = log(-log(w_decay)) = log(log(1+exp(-z)))  # softplus(-z)의 log
```
수치적으로 불안정해서 본 구현에서는 logit-space 가산형으로 대체. ablation 비교용으로만 별도 변형 권장.

### 11.4 Baseline 비교 프로토콜
공정한 비교를 위해:
1. `RWKV-v7/train_temp/`에서 baseline 동일 데이터/하이퍼파라미터로 학습
2. `RWKV-v7-kvw-decay/train_temp/`에서 동일하게 학습
3. 동일한 토큰 수 시점에서 loss/ppl 비교
4. 다운스트림 벡치 (`rwkv_mmlu_eval.py` 등)도 같은 토큰 수 시점에서 측정

KVW-Decay가 추가 파라미터(~+20%) 만큼 의미 있는 개선을 주는지가 핵심 질문 — 파라미터 동수 baseline (n_embd 살짝 키운 buy-back)과도 비교하면 더 정직.

---

## 12. 참고 파일

| 파일 | 용도 |
|---|---|
| `train.py` | 학습 entry point (argparse + Lightning Trainer) |
| `src/model.py` | **KVW-Decay 수정 위치** (`RWKV_Tmix_x070`) |
| `src/dataset.py` | binidx 데이터셋, magic_prime 검증 |
| `src/trainer.py` | LR schedule, 로그/체크포인트 콜백 |
| `src/binidx.py` | Megatron-style binidx 리더 |
| `cuda/*` | RWKV-7 fused CUDA kernels (수정 불필요) |
| `rwkv7_train_simplified.py` | (참고용) 간소화된 단일 파일 학습 데모 — 본 폴더와 별개 |
| `../../RWKV-v5/make_data.py` | JSONL → binidx 변환기 |
| `../../RWKV-v5/tokenizer/` | RWKV TRIE 토크나이저 |

---

## 13. 가중치 표 (참고)

KVW-Decay에서 추가/변경되는 파라미터만 발췌 (1.5B 기준 L24-D2048):

| name | shape | comment | initialization |
|---|---|---|---|
| `blocks.*.att.g_ww` | [3·2048, 2048] = [6144, 2048] | KVW-Decay (신규) | **0** |
| `blocks.*.att.b_ww` | [1, 1, 2048] | KVW-Decay (신규) | **0** |

나머지 파라미터는 baseline RWKV-7과 동일 (이전 README 표 그대로). `g_ww`/`b_ww`는 이름에 `.weight`가 없으므로:
- `configure_optimizers`: lr_1x 그룹 (weight decay 없음)
- `generate_init_weight`: 첫 분기로 흘러 zero 그대로 보존

이대로면 신규 파라미터에 weight decay가 안 걸리는 게 의도. wd가 필요하다고 판단되면 `model.py`의 `configure_optimizers`에서 `g_ww`만 `lr_decay`에 명시 추가.
