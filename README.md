# Pisto GPT 64M

 64M-parameter decoder only GPT model trained on TinyStories and instruction tuned on Alpaca + manual data.

## Weights

Download  instruction tuned weights from Hugging Face:


- **instruct_best.pt** (instruction-tuned) — [Download](https://huggingface.co/notpisto/pisto_gpt/resolve/main/wights/instruct_best.pt)

Place them in the `wights/` directory.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Use the menu to:

- `1` launch the web app
- `2` train the model, then choose `1` for pretraining or `2` for fine tuning

Open http://localhost:5000 after launching the app.

## Docker

```bash
docker build -t pisto-gpt .
docker run -p 5000:5000 pisto-gpt
```

## Training

From `run.py`, choose `2` for training, then:

- `1` for pretraining
- `2` for fine tuning

Direct commands:

### Pretraining

```bash
python train/pretraining.py
```
Downloads TinyStories, trains from scratch. Configure in `configs/train.json`.

### Instruction Tuning

```bash
python train/train_instruct.py
```
Loads pretrained weights from `wights/best.pt`, fine tunes on Alpaca + manual Q&A. Configure in `configs/instruct.json`.

Both scripts auto resume from checkpoints and support CPU/GPU.

## Model

| Param | Value |
|---|---|
| Architecture | GPT Decoder (Pre-LN) |
| Parameters | 64M |
| Layers | 10 |
| Heads | 8 |
| d_model | 720 |
| FFN | 2880 |
| Max length | 512 |
| Tokenizer | Byte-level (vocab 259) |
| Weight tying | Yes |

## Project Structure

```
├── configs/          # Training & generation configs
├── llm/              # Model definition & inference
├── train/            # Training scripts
├── ui/               # Flask web UI
└── wights/           # Trained checkpoints
```
