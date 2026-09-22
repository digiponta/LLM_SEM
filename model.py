# model.py
#
# CUDA/PyTorch implementation of the homemade v0.3 Transformer LM.

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Single-head causal self-attention, matching the original v0.3 design."""

    def __init__(self, d_model: int, causal: bool = True):
        super().__init__()
        self.d_model = d_model
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(float(self.d_model))

        if self.causal:
            time = x.size(1)
            mask = torch.triu(
                torch.ones(
                    (time, time),
                    dtype=torch.bool,
                    device=x.device,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        return self.out_proj(context)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return self-attention weights with shape [batch, query, key]."""
        q = self.q_proj(x)
        k = self.k_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(float(self.d_model))

        if self.causal:
            time = x.size(1)
            mask = torch.triu(
                torch.ones(
                    (time, time),
                    dtype=torch.bool,
                    device=x.device,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        return F.softmax(scores, dim=-1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        causal: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, causal=causal)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        hidden_dim: int = 256,
        causal: bool = True,
        context_length: int = 64,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if d_model <= 0 or num_layers <= 0 or hidden_dim <= 0:
            raise ValueError("Model dimensions must be > 0.")
        if context_length <= 0:
            raise ValueError("context_length must be > 0.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.context_length = context_length

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                hidden_dim=hidden_dim,
                causal=causal,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def config(self) -> Dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "causal": self.causal,
            "context_length": self.context_length,
        }

    def encode_hidden(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return contextual hidden states before vocabulary projection.

        Shape:
            token_ids: [batch, time]
            output   : [batch, time, d_model]
        """
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch, time].")

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def encode_semantic(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        hybrid_alpha: float = 0.5,
        normalize_hybrid: bool = False,
    ) -> torch.Tensor:
        """Return one semantic vector per input sequence.

        Supported pooling methods:
            mean      - mean of contextual token states
            last      - final token state
            bos       - first token state
            max       - element-wise maximum over token states
            attention - final-block attention-weighted token states
            hybrid    - alpha * attention + (1-alpha) * last
                         optionally L2-normalize both inputs and output

        Shape:
            token_ids      : [batch, time]
            attention_mask : [batch, time] (optional)
            output         : [batch, d_model]
        """
        supported = {"mean", "last", "bos", "max", "attention", "hybrid"}
        if pooling not in supported:
            raise ValueError(
                f"Unsupported semantic pooling: {pooling}. "
                f"Choose from {sorted(supported)}"
            )

        if not 0.0 <= hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be between 0.0 and 1.0.")

        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch, time].")

        if attention_mask is not None and attention_mask.shape != token_ids.shape:
            raise ValueError(
                "attention_mask must have the same [batch, time] shape "
                "as token_ids."
            )

        if pooling in {"attention", "hybrid"}:
            x = self.embedding(token_ids)
            final_attention = None

            for index, block in enumerate(self.blocks):
                if index == len(self.blocks) - 1:
                    normalized = block.norm1(x)
                    final_attention = block.attention.attention_weights(
                        normalized
                    )
                x = block(x)

            hidden = self.final_norm(x)
            assert final_attention is not None

            # Average over query positions to estimate how strongly each
            # token is used as a key by the final Transformer block.
            token_weights = final_attention.mean(dim=1)

            if attention_mask is not None:
                mask = attention_mask.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                token_weights = token_weights * mask

            token_weights = token_weights / (
                token_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            )
            attention_vector = (
                hidden * token_weights.unsqueeze(-1)
            ).sum(dim=1)

            if pooling == "attention":
                return attention_vector

            if attention_mask is None:
                last_vector = hidden[:, -1, :]
            else:
                lengths = attention_mask.to(
                    device=hidden.device,
                    dtype=torch.long,
                ).sum(dim=1).clamp_min(1)
                indices = lengths - 1
                batch_indices = torch.arange(
                    hidden.size(0),
                    device=hidden.device,
                )
                last_vector = hidden[batch_indices, indices, :]

            if normalize_hybrid:
                attention_vector = F.normalize(
                    attention_vector,
                    p=2,
                    dim=-1,
                )
                last_vector = F.normalize(
                    last_vector,
                    p=2,
                    dim=-1,
                )

            hybrid_vector = (
                hybrid_alpha * attention_vector
                + (1.0 - hybrid_alpha) * last_vector
            )

            if normalize_hybrid:
                hybrid_vector = F.normalize(
                    hybrid_vector,
                    p=2,
                    dim=-1,
                )

            return hybrid_vector

        hidden = self.encode_hidden(token_ids)

        if pooling == "bos":
            return hidden[:, 0, :]

        if pooling == "last":
            if attention_mask is None:
                return hidden[:, -1, :]

            lengths = attention_mask.to(
                device=hidden.device,
                dtype=torch.long,
            ).sum(dim=1).clamp_min(1)
            indices = lengths - 1
            batch_indices = torch.arange(
                hidden.size(0),
                device=hidden.device,
            )
            return hidden[batch_indices, indices, :]

        if pooling == "max":
            if attention_mask is None:
                return hidden.max(dim=1).values

            mask = attention_mask.to(
                device=hidden.device,
                dtype=torch.bool,
            ).unsqueeze(-1)
            masked = hidden.masked_fill(~mask, float("-inf"))
            return masked.max(dim=1).values

        # mean pooling
        if attention_mask is None:
            return hidden.mean(dim=1)

        mask = attention_mask.to(
            device=hidden.device,
            dtype=hidden.dtype,
        ).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denominator

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_hidden(token_ids)
        return self.lm_head(hidden)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save_checkpoint(
        self,
        filename: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
    ) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "format": "homemade-llm-gpu-v0.3",
            "config": self.config(),
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        filename: str,
        device: torch.device,
    ):
        checkpoint = torch.load(filename, map_location=device)
        config = checkpoint["config"]

        model = cls(
            vocab_size=int(config["vocab_size"]),
            d_model=int(config["d_model"]),
            num_layers=int(config["num_layers"]),
            hidden_dim=int(config["hidden_dim"]),
            causal=bool(config.get("causal", True)),
            context_length=int(config.get("context_length", 64)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint

    @torch.no_grad()
    def generate(
        self,
        token_ids: List[int],
        max_new_tokens: int = 100,
        eos_id: Optional[int] = None,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        repetition_penalty: float = 1.15,
    ) -> List[int]:
        if not token_ids:
            raise ValueError("token_ids must not be empty.")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0.")

        self.eval()
        device = next(self.parameters()).device
        generated = list(token_ids)

        for _ in range(max_new_tokens):
            context = generated[-self.context_length:]
            x = torch.tensor(
                [context],
                dtype=torch.long,
                device=device,
            )

            logits = self(x)[0, -1, :].clone()

            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if 0 <= token_id < logits.numel():
                        if logits[token_id] >= 0:
                            logits[token_id] /= repetition_penalty
                        else:
                            logits[token_id] *= repetition_penalty

            if temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                logits = logits / temperature

                if top_k is not None and 0 < top_k < logits.numel():
                    top_values, top_indices = torch.topk(logits, top_k)
                    probabilities = F.softmax(top_values, dim=-1)
                    selected = torch.multinomial(probabilities, num_samples=1)
                    next_id = int(top_indices[selected].item())
                else:
                    probabilities = F.softmax(logits, dim=-1)
                    next_id = int(
                        torch.multinomial(probabilities, num_samples=1).item()
                    )

            generated.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

        return generated
