# RWKV-7 `train_temp` — 상세 사용 가이드 (한국어)
# 2026-5-17
이 문서는 `RWKV-v7/train_temp/` 코드를 사용해 RWKV-7 ("Goose", x070) 모델을 처음부터 학습하는 전 과정을 설명합니다. 다루는 내용:

1. 환경 세팅
2. `train_temp` 폴더/파일 구조
3. 데이터셋(`.bin` / `.idx`) 준비 — v5의 `make_data.py` 사용
4. `magic_prime` 계산 방법
5. Stage 1 — 초기 가중치 생성(`demo-training-prepare.sh`)
6. Stage 3 — 실제 학습 실행(`demo-training-run.sh`)
7. 중단된 학습 이어하기(resume) / 베이스 모델에서 파인튜닝
8. 주요 CLI 플래그(전체 정리)
9. 멀티 GPU / 멀티 노드
10. VRAM 튜닝(`grad_cp`, `head_chunk`, `micro_bsz`, 커널 선택)
11. 로그, 체크포인트, `wandb`
12. 자주 발생하는 문제

---

## 1. 환경

검증된 조합 (참고: `requirements.txt`):

- Python **3.10+**
- PyTorch **2.5+** with CUDA **12.x** (cu121에만 한정되지 않으며, 최신 cu12 빌드면 됨)
- DeepSpeed (최신)
- `pytorch-lightning==1.9.5` ← **반드시 이 버전 고정**
- `ninja`, `wandb` (선택)

```bash
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade
```

기본 설정(L12-D768, ctx 512, micro_bsz 16)은 단일 GPU에서 약 7 GB VRAM이면 돌아갑니다. VRAM이 더 적다면 `micro_bsz`를 줄이거나 `grad_cp=1` / `head_chunk=4096`을 켜세요.

`cuda/` 폴더의 CUDA 커널들은 첫 실행 시 `torch.utils.cpp_extension.load`로 JIT 컴파일됩니다. "Loading extension…"에서 멈추면 `TORCH_EXTENSIONS_DIR`(기본 `~/.cache/torch_extensions/`)의 lock 파일을 지우고 다시 시도하세요.

---

## 2. 폴더 구조

```
RWKV-v7/train_temp/
├── README.md                         # 원본 짧은 README
├── requirements.txt
├── train.py                          # 진입점 (argparse + Lightning Trainer)
├── rwkv7_train_simplified.py         # 학습용 단일 파일 단순화 버전 (읽기 전용 참고)
├── demo-training-prepare.sh          # Stage 1 스크립트: rwkv-init.pth 생성 (MiniPile)
├── demo-training-run.sh              # Stage 3 스크립트: 실제 학습 (MiniPile)
├── demo-training-prepare-v7-pile.sh  # 풀 Pile 버전 (50304 vocab, ctx 4096)
├── demo-training-run-v7-pile.sh
├── src/
│   ├── model.py     # RWKV-7 모델 (init 스케줄, optimizer 그룹, 레이어 정의)
│   ├── trainer.py   # Lightning 콜백: LR 스케줄, 체크포인트 저장, 로깅
│   ├── dataset.py   # MyDataset: magic_prime으로 셔플하는 binidx 샘플러
│   └── binidx.py    # MMapIndexedDataset 리더 (Megatron과 동일 포맷)
└── cuda/            # WKV-7 / cmix / tmix / clampw / head-l2wrap CUDA 커널들
```

학습 결과물은 `out/L<N_LAYER>-D<N_EMBD>-<MODEL_TYPE>/`에 저장됩니다:

- `rwkv-init.pth` — 초기 가중치 (Stage 1 결과)
- `rwkv-<N>.pth` — 미니에포크 N 시점 체크포인트 (Stage 3, `--epoch_save` 주기)
- `rwkv-final.pth` — `my_exit_tokens` 도달 시 최종 체크포인트
- `train_log.txt` — 미니에포크 단위 loss / perplexity / LR / 타임스탬프

---

## 3. 데이터셋(`.bin` / `.idx`) 준비

트레이너는 `--data_type binidx`(Megatron 스타일 메모리 매핑 인덱스)만 받습니다. `--data_file`에는 **확장자 없는 prefix**를 넘기고, 내부적으로 `<prefix>.bin`과 `<prefix>.idx`를 읽습니다.

### 3a. 옵션 A — 미리 토크나이즈된 데이터셋 다운로드 (MiniPile, 가장 간단)

```bash
cd RWKV-v7/train_temp
mkdir -p data
wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

전체 토큰 수는 `1 498 226 207`이고 vocab은 RWKV world 토크나이저(`rwkv_vocab_v20230424.txt`)의 65536입니다. `demo-training-prepare.sh`의 기본값과 완벽히 일치합니다.

### 3b. 옵션 B — 내 JSONL로 `.bin` / `.idx` 직접 만들기 (v5의 `make_data.py` 사용)

`train_temp`에는 자체 토크나이저/빌더가 들어 있지 않습니다. 정식 도구는 `RWKV-v5/make_data.py`에 있습니다. 이 스크립트는 RWKV world 토크나이저를 쓰고, `src/binidx.py`가 읽는 것과 동일한 MMapIndexedDataset 포맷으로 저장합니다.

요구되는 JSONL 포맷 (한 줄에 하나의 문서, **빈 줄 금지**):

```
{"text": "첫 번째 문서 ..."}
{"text": "두 번째 문서 ..."}
```

실행:

```bash
cd /home/mechu/rwkv-lm/RWKV-v5
python make_data.py path/to/your.jsonl <N_EPOCH> <CTX_LEN>
```

- `<N_EPOCH>` — 데이터를 몇 번 셔플해서 이어붙일지. 사전학습용으로 데이터가 충분하다면 `1`로 충분, 파인튜닝용 작은 jsonl은 `3` 이상이 적당(샘플러가 다양한 시작점을 만들 수 있도록).
- `<CTX_LEN>` — 이 값에 맞는 `magic_prime`을 계산해 출력하는 용도.

이 스크립트가 처리하는 단계:

1. `your.jsonl`을 읽고 빈 줄 제거
2. `N_EPOCH`번 셔플 + 이어붙여 `make_data_temp.jsonl` 생성
3. 각 `"text"`를 `TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")`로 토크나이즈, 문서 끝에 토큰 `0` 추가
4. 현재 디렉토리에 `your.bin`, `your.idx` 작성
5. 일부 샘플을 다시 디코드해서 토크나이저 정확성 검증
6. **`magic_prime`** 출력 (= `tokens/CTX_LEN - 1` 이하의 가장 큰 `3n+2` 형 소수), 그리고 셸 스크립트에 그대로 붙여 넣을 수 있는 `--my_exit_tokens` / `--magic_prime` / `--ctx_len` 플래그 출력

결과 파일은 현재 디렉토리에 생기므로, 원하는 위치에서 실행하거나 이후에 옮기세요. `your.bin`, `your.idx`를 `RWKV-v7/train_temp/data/`로 옮긴 뒤 두 셸 스크립트에서 `--data_file "data/your"`로 지정합니다.

**vocab size:** world 토크나이저는 토큰 id가 약 65535까지 가므로 `--vocab_size 65536`을 씁니다(MiniPile과 동일). GPT-NeoX/Pile 토크나이저로 토크나이즈했다면 `--vocab_size 50304`(`demo-training-prepare-v7-pile.sh` 참고).

### 3c. 옵션 C — RWKV가 아닌 다른 토크나이저

포맷 자체는 그냥 MMapIndexedDataset(Megatron)입니다. 같은 스키마로 `.bin` / `.idx`를 만들어 주는 도구(예: Megatron-DeepSpeed 전처리기)면 무엇이든 동작합니다. dtype은 `src/binidx.py`에 선언된 것 중 하나여야 합니다(`make_data.py`는 `uint16` 사용 — vocab 65 535까지 안전).

---

## 4. `magic_prime` 계산

`src/dataset.py`의 샘플러는 step 인덱스를 파일 오프셋으로 결정론적으로 매핑합니다:

```
ii  = 1 + epoch * 40320 + idx * world_size + rank
i   = ((⌊magic_prime · (√5 − 1)/2⌋ · ii³) mod magic_prime) · ctx_len
```

이 매핑이 "에포크당 모든 청크를 거의 한 번씩 충돌 없이 방문"하려면 `magic_prime`이 다음을 만족해야 합니다:

- **소수**일 것,
- **`magic_prime % 3 == 2`** 일 것 (즉, 가장 큰 `3n+2` 형 소수),
- **`0.9 < magic_prime / (data_size // ctx_len) ≤ 1.0`** 일 것.

이 세 조건은 `MyDataset.__init__`(`src/dataset.py:42-44`)에서 `assert`로 체크됩니다. 하나라도 깨지면 학습이 즉시 중단됩니다.

### 4a. 가장 쉬운 방법 — `make_data.py`가 출력해 줌

`RWKV-v5/make_data.py`로 `.bin/.idx`를 만들었다면 마지막에 이미 다음 줄이 출력됩니다:

```
--my_exit_tokens <total_tokens> --magic_prime <N> --ctx_len <CTX_LEN>
```

그대로 복사해 넣으면 됩니다.

### 4b. 이미 있는 `.bin/.idx` — `compute_magic_prime.py` 사용

`RWKV-v5/compute_magic_prime.py`는 위 6번 단계만 따로 합니다:

```bash
cd /home/mechu/rwkv-lm/RWKV-v5
# 파일 상단의 DATA_NAME, CTX_LEN을 본인 경로/값으로 수정
python compute_magic_prime.py
```

`magic_prime`과 그대로 쓸 수 있는 CLI 플래그를 출력합니다.

### 4c. 수동 / 웹

`magic_prime`은 `data_size/ctx_len − 1` 이하의 가장 큰 `prime % 3 == 2` 소수입니다. 기본 MiniPile + ctx 512의 경우:

```
data_size/ctx_len − 1 = 1 498 226 207 / 512 − 1 ≈ 2 926 222.06
→ 이 값 이하의 가장 큰 3n+2 소수 = 2 926 181
```

https://www.dcode.fr/prime-numbers-search 에서 아래쪽으로 검색해도 됩니다. **`ctx_len`을 바꿀 때마다 `magic_prime`도 다시 계산해야 합니다.**

---

## 5. Stage 1 — `rwkv-init.pth` 생성

`demo-training-prepare.sh`는 `train.py`를 `--train_stage 1`로 실행합니다. 그러면 `src/trainer.py`의 `generate_init_weight`가 호출되고, `model.generate_init_weight()`에 정의된 파라미터별 init 스케줄(README 표 참고)대로 `rwkv-init.pth`를 `PROJ_DIR`에 저장한 뒤 종료합니다.

`demo-training-prepare.sh` 상단에서 설정할 변수:

| 변수            | 의미                                                                          |
|-----------------|-------------------------------------------------------------------------------|
| `MODEL_TYPE`    | `x070` (RWKV-7). 다른 변형 테스트가 아니면 그대로.                            |
| `N_LAYER`       | 깊이                                                                          |
| `N_EMBD`        | 폭 (= `dim_att`; FFN dim은 기본적으로 `3.5×N_EMBD`를 32단위로 반올림)        |
| `CTX_LEN`       | 학습 컨텍스트 길이. **계산한 `magic_prime`과 반드시 일치해야 함.**           |
| `PROJ_DIR`      | 출력 폴더. 기본값 `out/L${N_LAYER}-D${N_EMBD}-${MODEL_TYPE}` 그대로 추천.    |

`python train.py …` 라인 안에서도 아래를 본인 환경에 맞게 수정:

- `--data_file "data/<prefix>"` (확장자 없이)
- `--vocab_size` (world 토크나이저면 65536, Pile/NeoX면 50304)
- `--my_exit_tokens <binidx 총 토큰 수>`
- `--magic_prime <4단계에서 구한 값>`
- `--head_size 64` — x070은 64 고정 (CUDA 커널이 64로 하드코딩되어 있음)

실행:

```bash
cd RWKV-v7/train_temp
sh ./demo-training-prepare.sh
```

CUDA 커널 JIT 컴파일 → CPU에서 모델 빌드(`--accelerator cpu`) → `out/.../rwkv-init.pth` 저장 → 종료. 보통 몇 분이면 끝납니다.

> 팁: Stage 1은 `--load_model <path>`를 함께 주면 기존 체크포인트의 가중치를 새 초기화 모델에 **합쳐서** 저장할 수 있습니다(모양이 다르면 bilinear interpolation으로 리사이즈). `src/trainer.py:158`의 `generate_init_weight` 참고. 모델 크기 업스케일링에 유용합니다.

---

## 6. Stage 3 — 실제 학습

`demo-training-run.sh`는 `train.py`를 `--train_stage 3`로 실행합니다. 트레이너 동작:

1. `PROJ_DIR`에서 가장 큰 번호의 `rwkv-<N>.pth`를 찾아 로드(없으면 `rwkv-init.pth`)
2. `epoch_begin = max_p + 1`로 설정
3. `lr_init` → `lr_final`까지 코사인 감쇠(`my_exit_tokens` 토큰 동안, `warmup_steps`만큼 선형 워밍업)
4. **1 미니에포크 = 40320 샘플 = 40320 × ctx_len 토큰**, `epoch_save` 미니에포크마다 체크포인트 저장
5. `my_exit_tokens` 도달 시 `rwkv-final.pth` 저장 후 종료

스크립트 상단의 주요 노브:

| 변수         | 기본값   | 의미                                                                                                       |
|--------------|----------|------------------------------------------------------------------------------------------------------------|
| `M_BSZ`      | 16       | **GPU당** micro batch size. 실제 bsz = `num_nodes × devices × micro_bsz`. **`40320 / real_bsz`가 정수여야 함.** |
| `LR_INIT`    | 6e-4     | L12-D768은 6e-4, L24-D1024은 4e-4, L24-D2048은 3e-4. 파인튜닝은 다시 절반 이하로.                          |
| `LR_FINAL`   | 6e-5     | 코사인 타깃 (보통 `LR_INIT / 10`).                                                                         |
| `GRAD_CP`    | 1        | gradient checkpointing. 1 = VRAM 절약(느림), 0 = 빠름(VRAM 더 씀).                                         |
| `HEAD_CHUNK` | 0        | LM head 청킹. 0 = 빠름/VRAM 많이 씀, 4096 = 약 80% LM-head VRAM 절약(가장 느림), 65536 = 약 70% 절약.       |
| `KERNEL`     | `@rwkv3` | `""` = 기본 v1 커널, `@rwkv3` = H100과 일부 컨슈머 GPU에서 약 20% 빠른 신형 커널.                          |
| `EPOCH_SAVE` | 10       | N 미니에포크마다 체크포인트 저장.                                                                          |

> **주의:** `demo-training-run.sh` 상단의 `rm` 3줄은 이전 체크포인트(`rwkv-*0.pth`, `rwkv-71.pth`, `rwkv-final.pth`)를 삭제합니다. 클린 재시작 시 stale weights에서 잘못 resume되는 걸 막기 위한 장치입니다. **resume하고 싶다면 이 줄들을 주석 처리하세요.**

실행:

```bash
sh ./demo-training-run.sh
```

기대되는 초기 loss (이 값들에서 ±0.01 이내, 벗어나면 뭔가 잘못된 것):

```
0 4.875856 131.0863 ...
1 4.028621 56.1834  ...
2 3.801625 44.7739  ...
...
```

(2번째 열 = 평균 loss, 3번째 열 = perplexity = `exp(loss)`)

---

## 7. Resume / 파인튜닝

### 7a. Ctrl-C나 크래시 후 이어서 학습

`demo-training-run.sh`의 `rm` 줄들을 **주석 처리한 상태로** 다시 실행하기만 하면 됩니다. `train.py:111-132`의 Stage 3 로직이 자동으로:

- `PROJ_DIR`의 모든 `rwkv-*.pth` 나열
- 가장 큰 미니에포크 번호 선택(`init`은 -1로 취급)
- `args.load_model`을 그 파일로, `args.epoch_begin = max_p + 1`로 설정
- `warmup_steps`를 `<0`으로 둔 경우 10으로 자동 설정

가장 최근 체크포인트 로드가 실패하면(예: OOM kill로 파일 손상) 그 직전 체크포인트(`my_pile_prev_p`)로 자동 폴백합니다.

### 7b. 공개된 RWKV-7 체크포인트에서 파인튜닝

1. 체크포인트를 `PROJ_DIR`에 넣고 이름을 `rwkv-init.pth`로 바꿉니다(로더가 미니에포크 -1로 인식). **또는** `--load_model <path>`로 명시적으로 지정.
2. LR을 매우 작게: `--lr_init 1e-5 --lr_final 1e-5`.
3. 베이스와 동일한 `N_LAYER`, `N_EMBD`, `head_size`, `vocab_size`를 유지. 모양이 다르다면 Stage 1을 `--load_model`과 함께 써서 shape-adapting 복사(첫 축 방향 bilinear interpolation, `trainer.py:170-192`)를 수행하세요.
4. `--my_exit_tokens`를 파인튜닝 총 예산 토큰 수로 설정해 코사인 감쇠가 원하는 시점에 끝나도록.
5. 베이스의 vocab 구조가 다르면, 일치하는 토크나이저로 데이터를 다시 토크나이즈하세요.

체크포인트에 일부 키가 빠져 있다면 `--load_partial 1`을 줍니다 — 빠진 키는 새로 초기화된 가중치로 채워집니다.

### 7c. Stage 2 vs Stage 3

코드 상 `train_stage >= 2`이면 자동 resume 로직이 켜지고, `train_stage == 1`이면 초기화 가중치 생성 모드입니다. 제공된 스크립트는 1과 3을 씁니다. Resume 관점에서 `train_stage = 2`는 `3`과 사실상 동일하며, 원래는 pile-specific 모드 구분 흔적입니다. 그냥 스크립트의 값을 그대로 쓰세요.

---

## 8. 주요 CLI 플래그 정리

`train.py:18-58` 기준:

| 플래그                | 기본값       | 비고                                                                                          |
|-----------------------|--------------|-----------------------------------------------------------------------------------------------|
| `--load_model`        | `""`         | `.pth` 풀패스. Stage 3에서 `"0"`이면 resume 스캐너가 최신 파일을 자동 선택.                   |
| `--wandb`             | `""`         | wandb 프로젝트 이름; 빈 문자열이면 wandb 비활성화.                                            |
| `--proj_dir`          | `out`        | 체크포인트와 `train_log.txt` 저장 위치.                                                       |
| `--random_seed`       | `-1`         | `-1` = 글로벌 시드 미설정(멀티 GPU 샘플링 다양성을 위해 권장).                                |
| `--data_file`         |              | `.bin/.idx`의 prefix(확장자 없이).                                                            |
| `--data_type`         | `utf-8`      | 본 트레이너는 **반드시 `binidx`** (assert).                                                   |
| `--vocab_size`        | `0`          | 명시 설정(world면 65536, NeoX/Pile이면 50304).                                                |
| `--ctx_len`           | `1024`       | 계산한 `magic_prime`과 일치시킬 것.                                                           |
| `--epoch_steps`       | 자동 덮어씀  | `40320 // real_bsz`로 트레이너가 재계산.                                                      |
| `--epoch_count`       | 자동 덮어씀  | `magic_prime // 40320`로 트레이너가 재계산.                                                   |
| `--epoch_begin`       | `0`          | Stage 3 resume 시 자동 덮어씀.                                                                |
| `--epoch_save`        | `5`          | N 미니에포크마다 저장.                                                                        |
| `--micro_bsz`         | `12`         | GPU당 batch size.                                                                             |
| `--n_layer`           | `6`          | 깊이.                                                                                         |
| `--n_embd`            | `512`        | 폭.                                                                                           |
| `--dim_att`           | `0`          | `0`이면 `n_embd`로.                                                                           |
| `--dim_ffn`           | `0`          | `0`이면 `int((n_embd*3.5)//32*32)`로.                                                         |
| `--lr_init`           | `6e-4`       | 위의 스케일링 가이드 참고.                                                                    |
| `--lr_final`          | `1e-5`       | 코사인 타깃.                                                                                  |
| `--warmup_steps`      | `-1`         | `-1`이면 워밍업 없음; resume 시 10으로 자동 설정.                                             |
| `--beta1`/`--beta2`   | `0.9/0.99`   | Adam betas.                                                                                   |
| `--adam_eps`          | `1e-18`      | RWKV 안정성에 매우 작은 eps가 중요.                                                           |
| `--grad_cp`           | `0`          | 1 = gradient checkpointing(느리지만 VRAM 절약).                                               |
| `--weight_decay`      | `0`          | 소규모는 0.001, Pile 스케일은 0.1. **`model.py`에서 `wdecay`로 표시된 파라미터에만 적용됨.** |
| `--grad_clip`         | `1.0`        | loss 스파이크가 나는 경우 0.7/0.5/0.3/0.2로 줄여 보세요.                                      |
| `--train_stage`       | `0`          | `1` = init만, `3` = 학습(자동 resume).                                                        |
| `--ds_bucket_mb`      | `200`        | DeepSpeed bucket 크기. 컨슈머 GPU는 2, A100/H100은 200. 최신 DS에서 버그가 있어 기본 권장.   |
| `--head_size`         | `64`         | **x070은 64 고정** (CUDA 커널이 64로 하드코딩).                                               |
| `--head_chunk`        | `0`          | 0 = 빠름/VRAM 큼, 4096 = 최대 절약, 65536 = 중간.                                             |
| `--load_partial`      | `0`          | 1이면 빠진 키를 새 init으로 채움.                                                             |
| `--magic_prime`       | `0`          | 필수.                                                                                         |
| `--my_testing`        | `x070`       | 모델 변형 태그(model.py 코드 경로 분기).                                                      |
| `--kernel`            | `""`         | `@rwkv3` = 신형/더 빠른 커널.                                                                 |
| `--my_exit_tokens`    | `0`          | 총 토큰 수; 코사인 스케줄과 `rwkv-final.pth` 저장 시점을 결정.                                |

Pytorch-Lightning Trainer 플래그(`Trainer.add_argparse_args`로 전달):

- `--accelerator gpu|cpu`
- `--devices 1` (노드당 GPU 수)
- `--num_nodes 1`
- `--precision bf16` (권장; fp16은 오버플로우 위험, fp32는 매우 느림)
- `--strategy deepspeed_stage_2` (Stage 3도 가능하나 `RWKV_JIT_ON`이 꺼짐)
- `--enable_progress_bar True`

---

## 9. 멀티 GPU / 멀티 노드

`--devices <노드당_GPU수>`, `--num_nodes <N>` 설정. `real_bsz = num_nodes × devices × micro_bsz`가 **40320을 나누어 떨어져야 합니다.** 가능한 값: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 28, 30, 32, 35, 40, 42, …

DeepSpeed Stage 2가 기본이며 1.5B 모델까지는 일반 GPU에서 잘 동작합니다. 매우 큰 모델은 Stage 3로 전환(`--strategy deepspeed_stage_3`); 이 경우 JIT가 비활성화되고, 체크포인트 저장 방식도 달라집니다(`my_save`가 `trainer.save_checkpoint(..., weights_only=True)` 사용).

멀티 노드는 일반적인 `deepspeed --hostfile=… train.py …` 또는 `torchrun --nnodes=… --nproc_per_node=…` 래퍼로 실행하면 됩니다. 스크립트 자체는 `num_nodes`, `devices`만 정확히 설정되면 됩니다.

---

## 10. VRAM 튜닝

영향이 큰 순서대로:

1. **`micro_bsz`** — 활성화 메모리에 선형. OOM이면 가장 먼저 절반으로.
2. **`grad_cp=1`** — 블록별 gradient checkpointing. 활성화 메모리 30–50% 절약, 속도 약 20% 감소.
3. **`head_chunk`** — LM head loss 계산을 청크 단위로. LM head는 `vocab × n_embd` 크기라 vocab이 큰 소형 모델에서 차지 비율이 큼.
   - `0` = 청킹 없음, 가장 빠름, VRAM 최대.
   - `65536` = 청크 64k, LM head VRAM 약 70% 절약.
   - `4096` = 청크 4k, 약 80% 절약, 가장 느림.
4. **커널 선택** — `--kernel @rwkv3`은 H100 튜닝 clampw 커널; 다수의 컨슈머 GPU에서도 빠릅니다.
5. **DeepSpeed Stage 3** — optimizer/grad/param을 GPU 간 분산; 거대 모델용.
6. **Precision** — 학습은 bf16 사실상 유일. fp32는 동작하지만 약 3배 느림.

참고: `dim_att = n_embd`, `head_size = 64`이므로 head 수 = `n_embd / 64`. wkv7 CUDA 커널은 `T % CHUNK_LEN == 0`을 assert(`CHUNK_LEN = 16`)하므로 `ctx_len % 16 == 0`이어야 합니다.

---

## 11. 로그, 체크포인트, wandb

- **`train_log.txt`** — 미니에포크 종료마다 한 줄:
  `<epoch> <mean_loss> <perplexity> <lr> <timestamp> <current_epoch>`
- **체크포인트** — `rwkv-<N>.pth`(`epoch_save` 주기) 및 `rwkv-final.pth`(완료 시).
- **`wandb`** — `--wandb <project>`로 활성화; step마다 `loss`, `lr`, `wd`, `Gtokens`, `kt/s` 기록.

`epoch_count`는 `magic_prime // 40320`으로 자동 설정됩니다(= 데이터셋을 한 번 훑는 데 필요한 미니에포크 수). Lightning은 그 이후에도 계속 돕니다(`max_epochs = -1`)만, `my_exit_tokens` 분량의 토큰을 본 시점에 `rwkv-final.pth`를 저장하고 `exit(0)` 됩니다.

---

## 12. 자주 발생하는 문제

- **`assert is_prime(args.magic_prime)` 또는 비율 assert 실패** → `ctx_len`이나 `data_file`을 바꾸고 `magic_prime`을 다시 계산하지 않은 경우. `compute_magic_prime.py` 재실행.
- **`40320 % real_bsz != 0`** → `num_nodes × devices × micro_bsz`가 40320을 나누도록 `micro_bsz`를 조정.
- **"Loading extension rwkv7_…"에서 멈춤** → `~/.cache/torch_extensions/`의 stale lock 파일 때문. 해당 폴더 `rm -rf` 후 재실행.
- **첫 에포크 loss가 레퍼런스 ±0.01을 벗어남** → vocab size, 토크나이저, magic_prime, precision(fp16 쓰면 안 됨) 중 하나 잘못된 것. README의 기준 수치는 bf16 + MiniPile + 기본 설정 정확히 일치를 가정.
- **Resume이 엉뚱한 파일을 집음** → `PROJ_DIR`에 남아 있는 다른 `rwkv-*.pth`를 확인. 로더는 파일명의 가장 큰 정수를 선택.
- **첫 step에서 OOM** → `micro_bsz` 감소 → `grad_cp=1` → `head_chunk=4096` 순서로 시도. 컨슈머 GPU는 `--ds_bucket_mb 2`도 시도.
- **`pytorch_lightning` 버전 불일치** → 반드시 **1.9.5**. 다른 버전은 Trainer API 동작이 조용히 달라짐.
- **`head_size` 변경** → 권장하지 않음. 바꾸려면 CUDA 커널 내부의 하드코딩된 `64`와 `HEAD_SIZE`도 함께 수정해야 함. 모델은 import 시점에 `HEAD_SIZE == 64`를 assert.

---

## 빠른 레시피 — 내 데이터로 처음부터

```bash
# 1. binidx 만들기 (make_data.py가 v5 폴더에 있으므로 v5에서 실행)
cd /home/mechu/rwkv-lm/RWKV-v5
python make_data.py /path/to/mydata.jsonl 1 512
# 출력되는 --my_exit_tokens N --magic_prime P --ctx_len 512 라인을 기록.
mv mydata.bin mydata.idx /home/mechu/rwkv-lm/RWKV-v7/train_temp/data/

# 2. 두 셸 스크립트를 수정: --data_file "data/mydata",
#    --my_exit_tokens N, --magic_prime P, --ctx_len 512, --vocab_size 65536
cd /home/mechu/rwkv-lm/RWKV-v7/train_temp

# 3. 초기 가중치 생성
sh ./demo-training-prepare.sh

# 4. 학습 (나중에 resume할 거면 'rm' 줄들은 주석 처리)
sh ./demo-training-run.sh
```
