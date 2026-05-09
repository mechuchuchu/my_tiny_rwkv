# Quail Distributed Training — Design Document v3

> **Status**: Design Phase (2026-05-09)
> **Goal**: A distributed training system that enables community contributors with H100-class GPUs to collaboratively train RWKV-7 7B+ models

---

## 1. Motivation & Goals

### Why We're Building This

Large-scale LLM training is effectively monopolized by big tech. Without hundreds of A100/H100s and InfiniBand clusters, it's out of reach. Quail aims to tear down that barrier.

Key observations:
- The RWKV community already has many contributors with H100-class GPUs
- What if more participants naturally scales up training?
- What if a node dying doesn't stop training?
- What if anyone can participate over a regular home internet connection?

Solving all four simultaneously is the goal of this project.

### First Milestone

> Dozens of contributors with H100-class GPUs collaboratively training RWKV-7 7B.

### Long-term Goal

> BlinkDL: "I will upcycle it to MoE after I have the best dense"
> Best dense → MoE upcycle. v3 is the design for that dense phase.

---

## 2. Key Changes from v2

| Item | v2 | v3 |
|------|----|----|
| Weight format | 1-bit binary | **bf16** |
| Node VRAM | ~5-6GB | **~14GB** (H100 comfortable) |
| Server role | latent fp32 + Adam + quantization | **gradient averaging only** |
| Optimizer | centralized on server | **local on each node** |
| Gradient compression | bf16 raw | **DGC 99.9% sparse + mantissa 4-bit** |
| Communication (7B) | ~14GB/step | **~5MB/step** |
| Server RAM | 53+106GB required | **drastically reduced** |
| STE | required | **not needed** |

---

## 3. Core Ideas

### 3.1 bf16 Weight + Data Parallelism

Dropping 1-bit, redesigned around **bf16 + data parallelism**.

```
RWKV-7 7B bf16 weight:
  ~14GB → fits comfortably in H100 80GB
```

Every node holds the full model weights in bf16 and runs full forward/backward.
No need for STE, latent weights, quantization tricks, etc.

### 3.2 Local Optimizer

In v2, the server handled Adam steps. In v3, **each node runs Adam locally**.

```
Server role: compute averaged gradient, broadcast back
Node role:   receive averaged gradient → local Adam step → update weights
```

**Advantages:**
- Server computation drastically reduced (no 53+106GB Adam state)
- Each node manages its own Adam state independently
- Server needs no GPU — a lightweight CPU server is sufficient

**Note:**
- Since all nodes receive the **identical** averaged gradient every step, Adam states remain mathematically synchronized across existing nodes
- Only newly joining nodes need special handling (see Section 9.3)

### 3.3 Deep Gradient Compression (DGC) + Mantissa 4-bit

Two techniques combined to drastically reduce communication volume.

#### DGC (Lin et al., ICLR 2018)

99.9% of gradients are redundant → transmit only top-0.1%.

Four key techniques:
1. **Gradient Sparsification + Local Accumulation**: unsent gradients accumulated locally (no information loss)
2. **Momentum Correction**: corrects momentum discounting factor during sparse updates (critical)
3. **Momentum Factor Masking**: mask momentum buffer for transmitted gradients → prevents staleness
4. **Warm-up Training**: sparsity increases exponentially: 75%→93.75%→98.4375%→99.6%→99.9%

#### Mantissa 4-bit Quantization

Empirically confirmed (2026-05-09):
- **Continuous per-epoch bitflips on bf16 mantissa → cosine similarity = 1.0** (direction fully preserved)
- Training loss virtually identical
- Reason: directional information lives in the exponent; mantissa only adjusts magnitude precision

Therefore, reducing mantissa from 7-bit to 4-bit during gradient transmission preserves convergence direction.

Additional protection:
- **Gradient clipping**: removes large noise values at the source
- **Averaging**: N-node average → one node's noise diluted by 1/N

Practical noise impact: `noise_magnitude / (N_nodes × clip_threshold)`

#### Combined Effect (RWKV-7 7B)

```
Dense gradient (bf16):    ~14GB
DGC 99.9% sparse:         ~14MB
+ mantissa 4-bit:         ~5MB

→ ~5MB per step
→ Near-free communication on 1Gbps Ethernet
→ Home WiFi participation is feasible
```

Custom transmission format:
```
sign(1bit) + exponent(8bit) + mantissa(4bit) = 13bit per value
+ 16bit sparse index
→ 29 bits per nonzero gradient element
```

### 3.4 Layer-wise Gradient Sharding

All nodes run full forward/backward, but **each node only transmits gradients for its assigned layers**.

```
Node A: assigned layers 0~10   → transmits gradients for 10 layers
Node B: assigned layers 11~20  → transmits gradients for 11 layers
Node C: assigned layers 21~30  → transmits gradients for 10 layers
```

Per-node transmission after DGC:
```
(total ~5MB) / num_layers × assigned_layers
```
More nodes = less per-node transmission.

### 3.5 Behavior by Node Count

```
Few    (~10):    ~6 layers assigned per node
Medium (10~100): ~1-2 layers per node
Many   (100+):   multiple nodes share the same layers
                 averaged gradient quality improves (lower variance)
                 effectively H100-cluster-level throughput
```

---

## 4. Memory Requirements (RWKV-7 7B)

| Item | Size | Location |
|------|------|----------|
| bf16 weight | ~14GB | Node VRAM |
| Adam state (m+v, bf16) | ~28GB | Node VRAM or RAM |
| Activations (assigned layers) | ~few GB | Node VRAM |
| **Node VRAM total** | **~20~25GB** | H100 80GB ✓ |
| Gradient buffer (averaged) | ~5MB | Server RAM (transient) |
| Weight checkpoint | ~14GB | Server RAM |
| Gradient history (K=100) | ~500MB | Server RAM |
| **Server RAM total** | **~32GB** | Commodity CPU server ✓ |

> Offloading Adam state to CPU RAM reduces node VRAM pressure further.

### Minimum Server Spec (RWKV-7 7B)

```
CPU:  Decent multicore (for ZeroMQ handling)
RAM:  32GB (14GB weight + 500MB history + headroom)
GPU:  Not required
NIC:  1Gbps+ (50 nodes → ~250MB/step; 100+ nodes → 10Gbps recommended)
SSD:  For checkpoint storage
```

Training with an H100 cluster, yet the server is just a commodity machine.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Central Server                    │
│                                                     │
│  - Coordinator: node registration, layer assignment │
│  - GradientAggregator: sparse gradient averaging    │
│  - GradientHistory: history store + replay dispatch │
│  - Dispatcher: broadcast averaged gradient to nodes │
│  - Heartbeat Monitor: node liveness checking        │
│  (No GPU required, no large RAM required)           │
└─────────────────────────────────────────────────────┘
              ↕ ZeroMQ (pyzmq)
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Node A  │  │  Node B  │  │  Node C  │  ...
│ bf16     │  │ bf16     │  │ bf16     │
│ weight   │  │ weight   │  │ weight   │
│ (~14GB)  │  │ (~14GB)  │  │ (~14GB)  │
│          │  │          │  │          │
│ full     │  │ full     │  │ full     │
│ forward  │  │ forward  │  │ forward  │
│          │  │          │  │          │
│ layers   │  │ layers   │  │ layers   │
│ 0~10     │  │ 11~20    │  │ 21~30    │
│ grad     │  │ grad     │  │ grad     │
│ + Adam   │  │ + Adam   │  │ + Adam   │
└──────────┘  └──────────┘  └──────────┘
```

---

## 6. Training Loop (per N steps)

```
① Node join
   - Send REGISTER to server
   - Server: assign layer range + send current bf16 weight checkpoint

② N-step loop (node)
   for step in range(N):
       forward(full model, bf16 weight)
       compute loss
       backward(full model)
       accumulate assigned-layer gradients (DGC local accumulation)

③ Gradient compression + transmission (every N steps)
   DGC: select top-0.1% (hierarchical threshold)
   quantize mantissa to 4-bit
   node → server: sparse compressed gradient

④ Server: average + broadcast
   collect sparse gradients from nodes
   compute average (union support strategy)
   broadcast averaged gradient → all nodes

⑤ Node: local Adam step
   receive averaged gradient
   update DGC error feedback buffer
   Adam step → update bf16 weight
   apply momentum factor masking

⑥ repeat ②
```

---

## 7. DGC Implementation Details

### 7.1 Sparse Gradient Encoding

```python
# Transmission format: (value, index) pairs
# value: sign(1) + exponent(8) + mantissa(4) = 13bit → packed into 2 bytes
# index: 16-bit run-length encoding of zeros

def encode_sparse_gradient(grad: torch.Tensor, sparsity: float = 0.999):
    threshold = torch.quantile(grad.abs(), sparsity)
    mask = grad.abs() > threshold
    values = grad[mask]          # top-0.1% values
    indices = mask.nonzero()     # positions

    # quantize mantissa to 4 bits
    values_quantized = quantize_mantissa(values, bits=4)

    return values_quantized, indices
```

### 7.2 Momentum Correction

Must apply momentum correction — naive accumulation breaks convergence:

```python
# WRONG (convergence breaks):
# v_k = v_k + grad_k
# send sparse(v_k)

# CORRECT (DGC momentum correction):
u_k = momentum * u_k + grad_k       # update velocity
v_k = v_k + u_k                     # accumulate velocity
sparse_mask = |v_k| > threshold
send(v_k * sparse_mask)             # send sparse velocity
v_k = v_k * ~sparse_mask            # clear transmitted entries
u_k = u_k * ~sparse_mask            # momentum factor masking
```

### 7.3 Sparse Averaging (Server)

When sparse gradient supports differ across nodes:

- **Union**: include any value sent by any node; treat unsent positions as 0, then average
- **Intersection**: average only positions sent by all nodes (more conservative)

Default: **Union** (minimizes information loss)

---

## 8. Communication Protocol

### 8.1 ZeroMQ Patterns

| Communication | Pattern | Direction |
|---------------|---------|-----------|
| Node registration | REQ/REP | node → server |
| Sparse gradient upload | PUSH/PULL | node → server |
| Averaged gradient broadcast | PUB/SUB | server → node |
| Heartbeat | PUB/SUB | node → server |
| Layer reassignment | REQ/REP | server → node |

### 8.2 Message Format

```python
{
    "type": str,           # message type
    "step": int,           # current training step
    "layer_range": tuple,  # (start, end)
    "values": bytes,       # sparse gradient values (13-bit packed)
    "indices": bytes,      # sparse gradient indices (16-bit)
    "timestamp": float,
}
```

Message types:
```
REGISTER              # node registration request
REGISTER_ACK          # registration confirmed + layer assignment + checkpoint URL
SPARSE_GRADIENT       # DGC compressed gradient upload
AVERAGED_GRADIENT     # server → node averaged gradient broadcast
GRADIENT_HISTORY      # server → new node gradient history (for replay)
HEARTBEAT             # liveness signal
GOODBYE               # graceful shutdown
LAYER_REASSIGN        # layer reassignment on node failure
```

---

## 9. Fault Tolerance

### 9.1 Node Failure Detection
```
Heartbeat interval: 5 seconds
Timeout threshold:  15 seconds
Graceful shutdown:  GOODBYE message → handled immediately
```

### 9.2 Node Failure Handling
```
1. Server: identify failed node's layer range
2. Send LAYER_REASSIGN to surviving nodes
3. New nodes joining automatically fill vacant layer slots
4. New node: load latest weight checkpoint
```

No pipeline parallelism → no reforward needed.
Other nodes continue training uninterrupted.

### 9.3 New Node Adam State Synchronization — Gradient History Replay

Since all nodes receive the **identical** averaged gradient every step, **Adam states across existing nodes are always synchronized**.

Problem when a new node joins: Adam state starts from initialization → out of sync.

**Solution: Gradient History Replay**

Instead of transmitting a heavy Adam state snapshot, the server keeps a history of averaged gradients and lets the new node replay them locally.

```
Example: new node joins at step 558, last checkpoint = step 500

Server sends:
  1. step 500 weight checkpoint (bf16, ~14GB)
  2. averaged gradients for steps 500~557 — 58 entries
     (DGC compressed, ~5MB × 58 ≈ 290MB)

New node:
  load step 500 weights
  initialize Adam state (at step 500)
  run local Adam step × 58, replaying gradients in order
  → perfectly synchronized weight + Adam state at step 558
```

**What the server needs to keep:**
```
- weight checkpoint (refreshed every K steps): ~14GB × 1
- averaged gradient history (since last checkpoint): ~5MB × up to K entries
  K=100 → max ~500MB, fits entirely in RAM
```

**Advantages:**
- No need to transmit Adam state snapshot (~56GB)
- Gradient history is already compressed — lightweight
- New node achieves **mathematically identical** Adam state to existing nodes
- History size is bounded by checkpoint interval K

---

## 10. File Structure

```
quail_distributed/
├── README.md
├── DESIGN_v3.md               # this document
├── requirements.txt
│
├── server/
│   ├── __init__.py
│   ├── coordinator.py         # node registration, layer assignment
│   ├── gradient_agg.py        # sparse gradient averaging + broadcast
│   ├── gradient_history.py    # averaged gradient history + replay dispatch
│   └── heartbeat.py           # liveness monitoring
│
├── node/
│   ├── __init__.py
│   ├── worker.py              # main node loop
│   ├── forward_runner.py      # full forward/backward
│   ├── grad_extractor.py      # assigned-layer gradient extraction
│   ├── dgc.py                 # DGC: sparsification + momentum correction
│   └── local_optimizer.py     # local Adam + weight update
│
├── comm/
│   ├── __init__.py
│   ├── protocol.py            # message format definitions
│   ├── transport.py           # ZeroMQ wrapper
│   └── compress.py            # sparse encoding + mantissa 4-bit quantization
│
├── model/
│   ├── __init__.py
│   └── rwkv7.py               # RWKV-7 bf16 full model
│
├── launch.py                  # local multi-process test runner
└── join.py                    # contributor entry point
```

---

## 11. How to Participate (Target UX)

```bash
pip install quail-distributed

# Join training
python join.py --server quail.example.com --port 5555

# To leave
Ctrl+C  # SIGINT handler → sends GOODBYE
```

---

## 12. Implementation Priority

### Phase 1 — Local Prototype
- [ ] `rwkv7.py` bf16 full model forward/backward
- [ ] `dgc.py` DGC sparsification + momentum correction + warm-up
- [ ] `compress.py` mantissa 4-bit quantization + sparse encoding
- [ ] `protocol.py` message format
- [ ] `transport.py` ZeroMQ wrapper
- [ ] `launch.py` local multi-process test

### Phase 2 — Server/Node Basic Connection
- [ ] `coordinator.py` node registration + layer assignment
- [ ] `gradient_agg.py` sparse gradient averaging + broadcast
- [ ] `gradient_history.py` history store + replay on new node join
- [ ] `local_optimizer.py` local Adam step
- [ ] `worker.py` full forward + grad extraction + DGC + upload

### Phase 3 — Fault Tolerance
- [ ] Heartbeat + graceful shutdown
- [ ] Layer reassignment on failure
- [ ] New node join with gradient history replay

### Phase 4 — Community Release
- [ ] `join.py` contributor entry point
- [ ] Docker image
- [ ] Monitoring dashboard

---

## 13. Known Limitations & Open Questions

- **New node join latency**: replaying K gradient steps takes time; tuning K is a tradeoff between replay cost and checkpoint frequency
- **Sparse averaging strategy**: Union vs Intersection — which performs better needs empirical validation
- **DGC warm-up + large batch interaction**: optimal schedule when both are applied simultaneously is unverified
- **Malicious nodes**: gradient poisoning not yet addressed — proceeding on community trust for now
- **Server SPOF**: server failure halts all training → single server for now, redundancy deferred

---

## 14. Empirical Foundations

### Mantissa Bitflip Robustness (2026-05-09)

Continuous per-epoch bitflips on bf16 mantissa only:
- **Cosine similarity = 1.0** — direction fully preserved throughout
- Training loss virtually identical to baseline
- Weight change diverges slightly after epoch ~80 (when LR is small enough for noise to dominate), but no impact on convergence

Conclusion: aggressively quantizing mantissa to 4-bit during gradient transmission is safe.

### Large Batch Robustness

AdamW + cosine LR + gradient clipping eliminates sharp minima concerns at large batch sizes (experimentally confirmed).

---

## 15. Summary of Changes from v2

| Item | v2 | v3 |
|------|----|----|
| Weight format | 1-bit binary | **bf16** |
| STE | required | **not needed** |
| Node VRAM | ~5-6GB | **~14GB** |
| Server role | latent fp32 + Adam + quantization | **gradient averaging only** |
| Optimizer location | server | **each node locally** |
| Server RAM | 53+106GB | **~32GB** |
| Gradient compression | bf16 raw | **DGC 99.9% + mantissa 4-bit** |
| Communication (7B) | ~14GB/step | **~5MB/step** |
| Participation requirement | RTX 3070-class | **H100-class** (community standard) |

---

## 16. License

Apache 2.0

---

*Quail — "someday, RWKV-175B"*
