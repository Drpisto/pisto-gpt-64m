import sys, json, os
from pathlib import Path

_HERE = Path(__file__).parent
_PROJ = _HERE.parent
sys.path.insert(0, str(_PROJ / "llm"))

import torch as th
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from model import model as LLMModel

CONFIG_PATH = _PROJ / "configs" / "genert.json"
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

model_cfg = cfg["model"]
gen_cfg = cfg["generation"]
weight_path = str((CONFIG_PATH.parent / cfg["weight_path"]).resolve())

device = "cuda" if th.cuda.is_available() else "cpu"

mdl = LLMModel(
    d_model=model_cfg["d_model"],
    nhead=model_cfg["nhead"],
    dim_feedforward=model_cfg["dim_feedforward"],
    dropout=model_cfg["dropout"],
    transformer_layers=model_cfg["transformer_layers"],
    max_len=model_cfg["max_len"],
).to(device)
mdl.lm_head.weight = mdl.embed.weight

ckpt = th.load(weight_path, map_location=device, weights_only=False)
mdl.load_state_dict(ckpt["model"])
mdl.eval()
print(f"Loaded {weight_path} | step={ckpt['step']:,} | loss={ckpt['loss']:.4f}")

app = Flask(__name__)


def generate(
    prompt: str,
    max_new: int | None = None,
    temp: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    rep_pen: float | None = None,
) -> str:
    max_new = max_new if max_new is not None else gen_cfg["max_new"]
    temp = temp if temp is not None else gen_cfg["temp"]
    top_k = top_k if top_k is not None else gen_cfg["top_k"]
    top_p = top_p if top_p is not None else gen_cfg["top_p"]
    rep_pen = rep_pen if rep_pen is not None else gen_cfg["rep_pen"]

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
                logits[logits < top_vals[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sl, si = th.sort(logits, descending=True)
                cum = th.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cum - F.softmax(sl, dim=-1) >= top_p] = float("-inf")
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    try:
        response = generate(
            prompt,
            max_new=data.get("max_new"),
            temp=data.get("temp"),
            top_k=data.get("top_k"),
            top_p=data.get("top_p"),
            rep_pen=data.get("rep_pen"),
        )
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
