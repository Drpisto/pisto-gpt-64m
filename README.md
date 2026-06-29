# Pisto GPT 64M

This repo contains a 64M parameter decoder only GPT. It starts with TinyStories pretraining and then gets instruction tuned on Alpaca plus a small manual dataset.

## Weights

If you just want to try the model, download the checkpoints from Hugging Face and place them in `weights/`:

- `best.pt` for pretraining: [Download](https://huggingface.co/notpisto/pisto_gpt/resolve/main/weights/best.pt)
- `instruct_best.pt` for fine tuning: [Download](https://huggingface.co/notpisto/pisto_gpt/resolve/main/weights/instruct_best.pt)

If Hugging Face warns about unauthenticated downloads, export `HF_TOKEN` before you start training.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

`run.py` keeps things simple:

- `1` starts the web app
- `2` opens the training menu
  - `1` pretraining
  - `2` fine tuning

Once the app is running, open `http://localhost:5000`.

## Docker

```bash
docker build -t pisto-gpt .
docker run -p 5000:5000 pisto-gpt
```

## Training

If you prefer to run the scripts directly, use these:

### Pretraining

```bash
python train/pretraining.py
```

This trains from scratch on TinyStories. The settings live in `configs/train.json`.

### Fine Tuning

```bash
python train/train_instruct.py
```

This loads `weights/best.pt` and fine-tunes on Alpaca plus the manual Q&A data. The settings live in `configs/instruct.json`.

Both scripts resume from checkpoints automatically and work on CPU or GPU.

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
├── configs/          # Training and generation configs
├── llm/              # Model definition and inference
├── train/            # Training scripts
├── ui/               # Flask web UI
└── weights/           # Trained checkpoints
```
