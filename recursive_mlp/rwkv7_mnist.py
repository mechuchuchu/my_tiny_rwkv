import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math

# ── 설정 ──────────────────────────────────────────
WIDTH = 32
DEPTH = 256
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784  # MNIST 28×28
print(f"device: {DEVICE}")

# ── model ──────────────────────────────────────────
class RWKV7StyleMLP_Initialized(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        ddd = torch.linspace(0, 1, width).reshape(1, width)
        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7))
        self.receptance = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self._apply_rwkv7_init()

    def _apply_rwkv7_init(self):
        with torch.no_grad():
            nn.init.orthogonal_(self.receptance.weight, gain=0.5 / math.sqrt(self.width))
            nn.init.orthogonal_(self.key.weight, gain=0.05 / math.sqrt(self.width))
            nn.init.orthogonal_(self.value.weight, gain=0.5 / math.sqrt(self.width))
            nn.init.zeros_(self.output.weight)

    def forward(self, x, last_x):
        xx = last_x - x
        xr = x + xx * self.x_r
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        r, k, v = self.receptance(xr), self.key(xk), self.value(xv)
        state = k * v
        return self.output(torch.sigmoid(r) * state)


class RWKV7_FFN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.x_k = nn.Parameter(torch.ones(1, width) * 0.5)
        self.key = nn.Linear(width, width * 4, bias=False)
        self.value = nn.Linear(width * 4, width, bias=False)
        nn.init.zeros_(self.value.weight)
        nn.init.orthogonal_(self.key.weight, gain=math.sqrt(4))

    def forward(self, x, last_x):
        xx = last_x - x
        xk = x + xx * self.x_k
        k = torch.relu(self.key(xk)) ** 2
        return self.value(k)


class FullRWKV7RecursiveMLP(nn.Module):
    def __init__(self, width, depth, input_dim=784, num_classes=10):
        super().__init__()
        self.depth = depth
        self.input_proj = nn.Linear(input_dim, width)
        self.att = RWKV7StyleMLP_Initialized(width)
        self.ffn = RWKV7_FFN(width)
        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)
        self.res_scale = nn.Parameter(torch.ones(1) * (1.0 / math.sqrt(depth)))
        self.ln_out = nn.LayerNorm(width)  # nan 방지 핵심
        self.output = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.input_proj(x.view(x.size(0), -1))
        last_x = x
        for _ in range(self.depth):
            res = self.att(self.ln1(x), last_x)
            last_x = x
            x = x + res * self.res_scale
            res_ffn = self.ffn(self.ln2(x), last_x)
            x = x + res_ffn * self.res_scale
        x = self.ln_out(x)  # Recursion followed by scale normalization
        return self.output(x)


# ── data ────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── 학습 ──────────────────────────────────────────
model = FullRWKV7RecursiveMLP(WIDTH, DEPTH, input_dim=INPUT_DIM).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,}")
print(f"recursive core params: {WIDTH*WIDTH*4:,} (≈{WIDTH}×{WIDTH}×4)")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0)  # RWKV crashes if you apply weight decay to core parameters.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    # train
    model.train()
    train_loss = 0
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        if loss < 3:  # Skip explosion placement → Unintended regularization effect
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    # test
    model.eval()
    test_loss = 0
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            test_loss += criterion(out, labels).item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_train = train_loss / len(trainloader)
    avg_test  = test_loss  / len(testloader)
    acc = correct / total * 100
    print(f"ep{epoch+1:02d}: train={avg_train:.4f} test={avg_test:.4f} acc={acc:.2f}%")
