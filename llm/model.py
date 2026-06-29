import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformer import Transformer


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def __init__(self, path=None):
        if path is None:
            root = Path(__file__).parent.parent
            candidates = [root / "weights" / "bpe_tokenizer.json",
                          root / "configs" / "bpe_tokenizer.json"]
            path = str(next((p for p in candidates if p.exists()), candidates[-1]))
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)
        self.vocab_size = self._tok.get_vocab_size()

    def tokenize(self, text: str) -> list[int]:
        return [self.bos_id] + self._tok.encode(text).ids + [self.eos_id]

    def detokenize(self, token_ids: list[int]) -> str:
        ids = [t for t in token_ids if t not in (self.pad_id, self.bos_id, self.eos_id)]
        return self._tok.decode(ids)


class Model(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        transformer_layers=6,
        max_len=2048,
        vocab_size=None,
    ):
        super().__init__()
        self.tokenizer = ByteTokenizer()
        self.vocab_size = vocab_size or self.tokenizer.vocab_size
        self.max_len = max_len

        self.embed = nn.Embedding(self.vocab_size, d_model, padding_idx=self.tokenizer.pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.transformers = nn.ModuleList(
            [Transformer(d_model, nhead, dim_feedforward, dropout) for _ in range(transformer_layers)]
        )
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)

    def encode(self, text: str) -> th.Tensor:
        device = next(self.parameters()).device
        token_ids = self.tokenizer.tokenize(text)
        return th.tensor(token_ids, dtype=th.long, device=device).unsqueeze(0)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.detokenize(token_ids)

    def forward(self, input_ids: th.Tensor):
        input_ids = input_ids[:, -self.max_len :]
        x = self.embed(input_ids)
        positions = th.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        for block in self.transformers:
            x = block(x)

        return self.lm_head(x)

    def loss(self, input_ids: th.Tensor) -> th.Tensor:
        input_ids = input_ids[:, -self.max_len :]
        if input_ids.size(1) < 2:
            raise ValueError("input_ids sequence length must be >= 2")
        logits = self.forward(input_ids)
        logits = logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def train_step(self, input_ids: th.Tensor, optimizer: th.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        loss = self.loss(input_ids)
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return float(loss.detach().item())

    def train_step_text(self, text: str, optimizer: th.optim.Optimizer) -> float:
        return self.train_step(self.encode(text), optimizer)

    def save(self, path):
        th.save(self.state_dict(), path)

    def load(self, path):
        checkpoint = th.load(path, map_location="cpu")

        # If it's a training checkpoint dict, extract just the model weights
        if "model" in checkpoint:
            self.load_state_dict(checkpoint["model"])
        else:
            self.load_state_dict(checkpoint)
