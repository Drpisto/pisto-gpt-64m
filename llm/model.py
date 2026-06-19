import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformer import transformer


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    byte_offset = 3  # bytes 0..255 map to 3..258
    vocab_size = 259

    def tokenize(self, text: str) -> list[int]:
        data = text.encode("utf-8", errors="replace")
        return [self.bos_id] + [b + self.byte_offset for b in data] + [self.eos_id]

    def detokenize(self, token_ids: list[int]) -> str:
        bytes_out: list[int] = []
        for token_id in token_ids:
            if token_id in (self.pad_id, self.bos_id, self.eos_id):
                continue
            byte_val = token_id - self.byte_offset
            if 0 <= byte_val <= 255:
                bytes_out.append(byte_val)
        return bytes(bytes_out).decode("utf-8", errors="replace")


class model(nn.Module):
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
            [transformer(d_model, nhead, dim_feedforward, dropout) for _ in range(transformer_layers)]
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
        # تقليص التدرج لتجنب انفجاره
        th.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return float(loss.detach().item())

    def train_step_text(self, text: str, optimizer: th.optim.Optimizer) -> float:
        return self.train_step(self.encode(text), optimizer)

    @th.no_grad()
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> str:
        """
        توليد نص مع التحكم في العشوائية.

        المعاملات:
            temperature : قيمة أعلى = نص أكثر إبداعاً، قيمة أقل = نص أكثر تركيزاً
                          القيم المقترحة: 0.5 (دقيق) .. 1.2 (إبداعي)
            top_k       : خذ فقط أفضل k توكن احتمالاً (0 = معطَّل)
            top_p       : nucleus sampling — خذ أصغر مجموعة احتمالها التراكمي >= top_p
                          (1.0 = معطَّل)
        """
        self.eval()
        input_ids = self.encode(input_text)
        generated_tokens = []

        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            logits = logits[:, -1, :]  # آخر توكن فقط  [1, vocab_size]

            # --- Temperature ---
            if temperature != 1.0:
                logits = logits / temperature

            # --- Top-K ---
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                top_values, _ = th.topk(logits, k)
                min_top = top_values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_top, float("-inf"))

            # --- Top-P (Nucleus) ---
            if top_p < 1.0:
                sorted_logits, sorted_indices = th.sort(logits, descending=True)
                cumulative_probs = th.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # أزل التوكنات التي تتجاوز الحد التراكمي
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                logits = th.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

            # --- Sampling ---
            probs = F.softmax(logits, dim=-1)
            next_token_id = th.multinomial(probs, num_samples=1)  # [1, 1]

            # توقف إذا وصلنا لتوكن النهاية
            if next_token_id.item() == self.tokenizer.eos_id:
                break

            generated_tokens.append(next_token_id.item())
            input_ids = th.cat([input_ids, next_token_id], dim=1)
            
            # Stop if we hit a user turn marker or end marker (avoid repeating conversation pattern)
            if len(generated_tokens) >= 3:
                recent_bytes = [t - self.tokenizer.byte_offset for t in generated_tokens[-5:] if self.tokenizer.byte_offset <= t < self.tokenizer.byte_offset + 256]
                try:
                    recent_text = bytes(recent_bytes).decode("utf-8", errors="ignore").lower()
                    if "<|end|>" in recent_text or "<|user|>" in recent_text:
                        break
                except:
                    pass

        return self.decode(input_ids.squeeze(0).tolist())

    def save(self, path):
        th.save(self.state_dict(), path)

    def load(self, path):
        checkpoint = th.load(path, map_location="cpu")

        # If it's a training checkpoint dict, extract just the model weights
        if "model" in checkpoint:
            self.load_state_dict(checkpoint["model"])
        else:
            self.load_state_dict(checkpoint)
