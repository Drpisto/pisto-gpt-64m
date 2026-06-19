# ============================================================
# Cell 1: Install
# ============================================================
!pip install -q datasets

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

SAVE_DIR = Path("/content/drive/MyDrive/llm_v2_instruct")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {th.cuda.get_device_name(0)}")

# ============================================================
# Cell 3: Model + تحميل pretrained
# ============================================================
mdl = LLMModel(
    d_model=720, nhead=8, dim_feedforward=2880,
    dropout=0.05, transformer_layers=10, max_len=512,
).to(device)
mdl.lm_head.weight = mdl.embed.weight

ckpt = th.load(
    "/content/drive/MyDrive/llm_v2/best.pt",
    map_location=device, weights_only=False
)
mdl.load_state_dict(ckpt["model"])
print(f"Loaded pretrained ✓ | tokens={ckpt.get('step',0)*16*512*2/1e6:.0f}M")
print(f"Parameters: {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M")

# ============================================================
# Cell 4: Dataset
# ============================================================
from datasets import load_dataset

PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

# بيانات يدوية بسيطة
manual_data = [
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi there!", "Hi! What can I do for you?"),
    ("How are you?", "I am doing well, thank you for asking!"),
    ("Good morning!", "Good morning! Hope you have a great day!"),
    ("Good night!", "Good night! Sleep well!"),
    ("What is your name?", "I am a small language model. How can I help?"),
    ("Who are you?", "I am an AI assistant. I am here to help you!"),
    ("What can you do?", "I can answer simple questions and tell short stories!"),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("What is 5 + 3?", "5 + 3 equals 8."),
    ("What is 10 - 4?", "10 - 4 equals 6."),
    ("What is 3 times 3?", "3 times 3 equals 9."),
    ("What is 10 divided by 2?", "10 divided by 2 equals 5."),
    ("What color is the sky?", "The sky is blue on a clear day."),
    ("What color is the sun?", "The sun appears yellow or orange."),
    ("What color is grass?", "Grass is green."),
    ("What is water?", "Water is a clear liquid. We drink it every day."),
    ("What is fire?", "Fire is hot and gives light. It can be dangerous."),
    ("What is the sun?", "The sun is a big star. It gives us light and warmth."),
    ("What is the moon?", "The moon is a large rock that goes around the Earth."),
    ("What is rain?", "Rain is water that falls from clouds in the sky."),
    ("What is snow?", "Snow is frozen water that falls from the sky in winter."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of the USA?", "The capital of the USA is Washington D.C."),
    ("What is the biggest ocean?", "The Pacific Ocean is the biggest ocean."),
    ("What is the biggest animal?", "The blue whale is the biggest animal in the world."),
    ("Tell me a short story.", "Once there was a little dog named Max. He loved to run and play in the park every day. One day he found a ball and was very happy."),
    ("Tell me a story about a cat.", "There was a small cat named Luna. She liked to sit by the window and watch the birds fly by. Every night she slept by the fire."),
    ("Tell me a story about a boy.", "There was a boy named Tim. He liked to read books and play outside. One day he found a small bird and took care of it."),
    ("Tell me a story about a girl.", "There was a girl named Lily. She loved flowers and liked to help her mom in the garden. She was very kind to everyone."),
    ("Tell me a story about a dragon.", "Once there was a small dragon who could not fly. He practiced every day. One morning he jumped and flew high in the sky."),
    ("Is the sky blue?", "Yes, the sky is blue on a clear day."),
    ("Is fire cold?", "No, fire is very hot."),
    ("Can fish swim?", "Yes, fish can swim very well."),
    ("Can dogs fly?", "No, dogs cannot fly."),
    ("Is water wet?", "Yes, water is wet."),
    ("What is a dog?", "A dog is an animal. Dogs are friendly and loyal pets."),
    ("What is a cat?", "A cat is a small furry animal. Cats are popular pets."),
    ("What is a tree?", "A tree is a tall plant with a trunk and leaves."),
    ("What is a book?", "A book has pages with words. We read books to learn and have fun."),
    ("What is a car?", "A car is a vehicle with four wheels. People use cars to travel."),
]

# Alpaca مفلتر — أسئلة قصيرة وبسيطة فقط
print("Loading Alpaca...")
ds = load_dataset("tatsu-lab/alpaca", split="train")
alpaca = [
    (s["instruction"], s["output"])
    for s in ds
    if not s["input"]
    and len(s["output"]) < 150
    and len(s["instruction"]) < 80
    and len(s["output"]) > 15
]
random.shuffle(alpaca)
alpaca = alpaca[:2000]
print(f"Alpaca filtered: {len(alpaca):,}")

all_data = manual_data * 10 + alpaca  # كرر اليدوي 10x لأهميته
random.shuffle(all_data)
print(f"Total: {len(all_data):,} samples")

class InstructDataset(Dataset):
    def __init__(self, samples, tokenizer, seq_len=512):
        self.seq_len = seq_len
        print(f"Tokenizing {len(samples):,} samples...")
        all_tokens = []
        for inst, resp in samples:
            text = PROMPT.format(instruction=inst, response=resp)
            all_tokens.extend(tokenizer.tokenize(text))
        self.data = th.tensor(all_tokens, dtype=th.long)
        self.n = (len(self.data) - 1) // seq_len
        print(f"Tokens: {len(self.data):,} → {self.n:,} chunks")

    def __len__(self): return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            pad = th.zeros(self.seq_len + 1 - len(chunk), dtype=th.long)
            chunk = th.cat([chunk, pad])
        return chunk

tokenizer = ByteTokenizer()
split = int(len(all_data) * 0.95)
train_ds = InstructDataset(all_data[:split], tokenizer)
eval_ds  = InstructDataset(all_data[split:], tokenizer)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
eval_loader  = DataLoader(eval_ds,  batch_size=8, shuffle=False,
                          num_workers=2, pin_memory=True, drop_last=True)
print("Dataset ready ✓")

# ============================================================
# Cell 5: Instruction Tuning
# ============================================================
MAX_HOURS    = 4.5
GRAD_ACCUM   = 2
MAX_STEPS    = 100_000
WARMUP_STEPS = 100
LOG_EVERY    = 20
EVAL_EVERY   = 200
LR           = 5e-5
LR_MIN       = 5e-6

decay, no_decay = [], []
for name, p in mdl.named_parameters():
    if not p.requires_grad: continue
    if p.dim() < 2 or any(x in name for x in ["ln", "bias", "embed"]):
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = th.optim.AdamW([
    {"params": decay,    "weight_decay": 0.01},
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
                y.reshape(-1), ignore_index=0,
            )
        total += loss.item()
    mdl.train()
    return total / max(n, 1)

def save_ckpt(step, loss):
    th.save({
        "step": step, "loss": loss,
        "model": mdl.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, SAVE_DIR / "instruct_best.pt")
    print(f"  ✓ instruct_best.pt (step={step:,})")

# Resume
step = 0
best_eval = float("inf")
loss_val = 0.0

instruct_ckpt = SAVE_DIR / "instruct_best.pt"
if instruct_ckpt.exists():
    ckpt = th.load(instruct_ckpt, map_location=device, weights_only=False)
    mdl.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    print(f"Resumed instruct: step={step:,}")
else:
    print("Starting Instruction Tuning fresh")

log_path = SAVE_DIR / "instruct_log.jsonl"
t0 = time.time()
mdl.train()

print(f"\n{'='*50}")
print(f"Instruction Tuning | max {MAX_HOURS}h | lr={LR}")
print(f"{'='*50}\n")

while step < MAX_STEPS:
    for batch in train_loader:

        elapsed_h = (time.time() - t0) / 3600
        if elapsed_h >= MAX_HOURS:
            save_ckpt(step, loss_val)
            print(f"\nDone. step={step:,} | best_eval={best_eval:.4f}")
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
                    y.reshape(-1), ignore_index=0,
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
            ppl = math.exp(min(loss_val, 20))
            print(f"step={step:5d} | loss={loss_val:.4f} | ppl={ppl:.1f} | "
                  f"lr={lr:.2e} | gnorm={gnorm:.2f} | {elapsed_h:.2f}h")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "loss": loss_val, "ppl": ppl,
                    "lr": lr, "gnorm": float(gnorm),
                    "elapsed_h": elapsed_h,
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

# ============================================================
# Cell 6: Chat — بعد التدريب
# ============================================================
ckpt = th.load(
    str(SAVE_DIR / "instruct_best.pt"),
    map_location=device, weights_only=False,
)
mdl.load_state_dict(ckpt["model"])
mdl.eval()
print("Loaded ✓")

def chat(prompt, max_new=100, temp=0.5, top_k=20, top_p=0.9, rep_pen=1.3):
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    input_ids = th.tensor(
        [mdl.tokenizer.tokenize(formatted)], dtype=th.long, device=device
    )
    generated = []

    with th.no_grad():
        for _ in range(max_new):
            logits = mdl(input_ids)[:, -1, :]
            if rep_pen != 1.0 and generated:
                for tid in set(generated):
                    if logits[0, tid] > 0: logits[0, tid] /= rep_pen
                    else: logits[0, tid] *= rep_pen
            logits = logits / temp
            if top_k > 0:
                top_vals, _ = th.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float('-inf')
            if top_p < 1.0:
                sl, si = th.sort(logits, descending=True)
                cum = th.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cum - F.softmax(sl, dim=-1) >= top_p] = float('-inf')
                logits = th.zeros_like(logits).scatter_(1, si, sl)
            next_id = th.multinomial(F.softmax(logits, dim=-1), 1).item()
            if next_id == mdl.tokenizer.eos_id:
                break
            generated.append(next_id)
            input_ids = th.cat([
                input_ids, th.tensor([[next_id]], device=device)
            ], dim=1)
            if len(generated) >= 5:
                recent = mdl.tokenizer.detokenize(generated[-8:])
                if "###" in recent:
                    generated = generated[:-8]
                    break

    return mdl.tokenizer.detokenize(generated).strip()

# اختبر
tests = [
    "Hello!",
    "How are you?",
    "What is 2 + 2?",
    "What color is the sky?",
    "What is the capital of France?",
    "Tell me a short story.",
    "Can dogs fly?",
    "What is the sun?",
]

print("\n" + "="*50)
for q in tests:
    print(f"Q: {q}")
    print(f"A: {chat(q)}")
    print("-"*35)
