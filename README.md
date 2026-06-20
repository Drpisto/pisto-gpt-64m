# Pisto GPT 64M

A from-scratch 64M-parameter decoder-only GPT model trained on TinyStories and instruction-tuned on Alpaca + manual data.

## Quick Start

```bash
pip install flask torch
python ui/app.py
```

Open http://localhost:5000

## Docker

```bash
docker build -t pisto-gpt .
docker run -p 5000:5000 pisto-gpt
```

## Training

### Pretraining

```bash
pip install datasets
python train/pretraining.py
```
Downloads TinyStories, trains from scratch. Configure in `configs/train.json`.

### Instruction Tuning

```bash
python train/train_instruct.py
```
Loads pretrained weights from `wights/best.pt`, fine-tunes on Alpaca + manual Q&A. Configure in `configs/instruct.json`.

Both scripts auto-resume from checkpoints and support CPU/GPU.

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
