import torch as th
import torch.nn.functional as F
from model import model as LLMModel

device = "cuda" if th.cuda.is_available() else "cpu"

_loaded_models = {}  


def load_model(weight_path):
    if weight_path in _loaded_models:
        return _loaded_models[weight_path]

    mdl = LLMModel(d_model=720, nhead=8, dim_feedforward=2880, dropout=0.0, transformer_layers=10, max_len=512).to(device)
    mdl.lm_head.weight = mdl.embed.weight

    ckpt = th.load(weight_path, map_location=device, weights_only=False)
    mdl.load_state_dict(ckpt["model"])
    mdl.eval()
    print(f"Loaded ✓ | {weight_path} | step={ckpt['step']:,} | loss={ckpt['loss']:.4f}")

    _loaded_models[weight_path] = mdl
    return mdl


def chat(prompt, weight_path="/home/katcho/Desktop/test/llm with pytorch training/test wights/instruct_best(1).pt",
          max_new=100, temp=0.5, top_k=20, top_p=0.9, rep_pen=1.3):

    mdl = load_model(weight_path)

    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    tokens = mdl.tokenizer.tokenize(formatted)
    if tokens[-1] == mdl.tokenizer.eos_id:
        tokens = tokens[:-1]

    input_ids = th.tensor([tokens], dtype=th.long, device=device)
    generated = []

    with th.no_grad():
        for _ in range(max_new):
            logits = mdl(input_ids)[:, -1, :]

            if rep_pen != 1.0 and generated:
                for tid in set(generated):
                    logits[0, tid] /= rep_pen if logits[0, tid] > 0 else 1 / rep_pen

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
            input_ids = th.cat([input_ids, th.tensor([[next_id]], device=device)], dim=1)

            if len(generated) >= 5 and "###" in mdl.tokenizer.detokenize(generated[-8:]):
                generated = generated[:-8]
                break

    return mdl.tokenizer.detokenize(generated).strip()


# اختبر — مع path اختياري
tests = [
    
     "Tell me a short story."
    
]

print("\n" + "=" * 50)
for q in tests:
    print(f"Q: {q}")
    print(f"A: {chat(q)}")     
    print("-" * 35)