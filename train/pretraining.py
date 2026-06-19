# ============================================================
# Cell 1: Install
# ============================================================


# ============================================================
# Cell 2: Imports
# ============================================================
import sys, os, math, time, json, random
from pathlib import Path

import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/content")
from model import model as LLMModel, ByteTokenizer

SAVE_DIR = Path("/content/drive/MyDrive/llm_v2")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {th.cuda.get_device_name(0)}")
print(f"Save dir: {SAVE_DIR}")

# ============================================================
# Cell 3: Model
# ============================================================
mdl = LLMModel(
    d_model=720,
    nhead=8,
    dim_feedforward=2880,
    dropout=0.1,
    transformer_layers=10,
    max_len=512,
).to(device)

mdl.lm_head.weight = mdl.embed.weight

total = sum(p.numel() for p in mdl.parameters())
print(f"Parameters: {total/1e6:.1f}M")

# ============================================================
# Cell 4: Dataset — TinyStories
# ============================================================
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=512):
        self.seq_len = seq_len
        print(f"Tokenizing {len(texts):,} docs...")
        all_tokens = []
        for text in texts:
            all_tokens.extend(tokenizer.tokenize(text))
        self.data = th.tensor(all_tokens, dtype=th.long)
        self.n = (len(self.data) - 1) // seq_len
        print(f"Tokens: {len(self.data):,} → {self.n:,} chunks")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            pad = th.zeros(self.seq_len + 1 - len(chunk), dtype=th.long)
            chunk = th.cat([chunk, pad])
        return chunk

tokenizer = ByteTokenizer()

print("Loading TinyStories...")
ds = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True,
)

texts = []
for i, s in enumerate(ds):
    if i >= 200_000: break
    text = s.get("text", "")
    if len(text) > 50:
        texts.append(text)
    if i % 20_000 == 0:
        print(f"  {i:,} stories...")

random.shuffle(texts)
split = int(len(texts) * 0.95)

train_ds = TextDataset(texts[:split], tokenizer)
eval_ds  = TextDataset(texts[split:], tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
eval_loader  = DataLoader(eval_ds,  batch_size=16, shuffle=False,
                          num_workers=2, pin_memory=True, drop_last=True)

print("Dataset ready ✓")

# ============================================================
# Cell 5: Train
# ============================================================
MAX_HOURS    = 4.5
GRAD_ACCUM   = 2
MAX_STEPS    = 200_000
WARMUP_STEPS = 500
LOG_EVERY    = 20
EVAL_EVERY   = 200
LR           = 3e-4
LR_MIN       = 3e-5

# Optimizer
decay, no_decay = [], []
for name, p in mdl.named_parameters():
    if not p.requires_grad: continue
    if p.dim() < 2 or any(x in name for x in ["ln", "bias", "embed"]):
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = th.optim.AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=LR, betas=(0.9, 0.95), fused=True)

scaler = th.cuda.amp.GradScaler()

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return LR_MIN + (LR - LR_MIN) * cosine

@th.no_grad()
def evaluate(n=30):
    mdl.eval()
    total = 0
    for i, batch in enumerate(eval_loader):
        if i >= n: break
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        with th.cuda.amp.autocast():
            loss = F.cross_entropy(
                mdl(x).reshape(-1, mdl.vocab_size),
                y.reshape(-1),
                ignore_index=0,
            )
        total += loss.item()
    mdl.train()
    return total / max(n, 1)

def save_ckpt(step, loss):
    path = SAVE_DIR / "best.pt"
    th.save({
        "step": step,
        "loss": loss,
        "model": mdl.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  ✓ best.pt (step={step:,})")

# ── Resume ──
step      = 0
best_eval = float("inf")
loss_val  = 0.0

drive_best = SAVE_DIR / "best.pt"
local_best = Path("/content/best.pt")

if drive_best.exists():
    resume_path = drive_best
elif local_best.exists():
    resume_path = local_best
else:
    resume_path = None

if resume_path:
    print(f"Loading: {resume_path}...")
    ckpt = th.load(resume_path, map_location=device, weights_only=False)
    mdl.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    print(f"Resumed: step={step:,} | tokens={step*16*512*2/1e6:.0f}M")
else:
    print("Starting fresh")

# ── Loop ──
log_path = SAVE_DIR / "log.jsonl"
t0       = time.time()
mdl.train()

print(f"\n{'='*50}")
print(f"Pretraining | max {MAX_HOURS}h | target ~500M tok")
print(f"{'='*50}\n")

while step < MAX_STEPS:
    for batch in train_loader:

        elapsed_h = (time.time() - t0) / 3600
        if elapsed_h >= MAX_HOURS:
            save_ckpt(step, loss_val)
            tokens_m = step * 16 * 512 * GRAD_ACCUM / 1e6
            print(f"\nDone. step={step:,} | tokens={tokens_m:.0f}M | best_eval={best_eval:.4f}")
            break

        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0

        for _ in range(GRAD_ACCUM):
            x = batch[:, :-1].to(device, non_blocking=True)
            y = batch[:, 1:].to(device, non_blocking=True)
            with th.cuda.amp.autocast():
                loss = F.cross_entropy(
                    mdl(x).reshape(-1, mdl.vocab_size),
                    y.reshape(-1),
                    ignore_index=0,
                ) / GRAD_ACCUM
            scaler.scale(loss).backward()
            loss_val += loss.item()

        scaler.unscale_(optimizer)
        gnorm = th.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        if step % LOG_EVERY == 0:
            elapsed_h = (time.time() - t0) / 3600
            tokens_m  = step * 16 * 512 * GRAD_ACCUM / 1e6
            ppl       = math.exp(min(loss_val, 20))
            print(f"step={step:6d} | loss={loss_val:.4f} | ppl={ppl:.1f} | "
                  f"lr={lr:.2e} | gnorm={gnorm:.2f} | "
                  f"{elapsed_h:.2f}h | {tokens_m:.0f}M tok")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "loss": loss_val, "ppl": ppl,
                    "lr": lr, "gnorm": float(gnorm),
                    "elapsed_h": elapsed_h, "tokens_M": tokens_m,
                }) + "\n")

        if step % EVAL_EVERY == 0:
            ev = evaluate()
            ppl = math.exp(min(ev, 20))
            print(f"  ↳ eval={ev:.4f} | ppl={ppl:.1f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "eval_loss": ev}) + "\n")
            if ev < best_eval:
                best_eval = ev
                save_ckpt(step, ev)