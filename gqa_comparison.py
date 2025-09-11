import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from typing import Optional, Dict, Any
import time

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import flax.linen as fnn
from functools import partial
import numpy as np


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


class LightningGQABlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 2880,
            num_query_heads: int = 64,
            num_kv_heads: int = 8,
            window_size: int = 128,
            layer_idx: int = 0,
            dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_query_groups = num_query_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_banded = (layer_idx % 2 == 1)
        self.window_size = window_size

        self.qkv_proj = nn.Linear(
            embed_dim,
            embed_dim + 2 * self.num_kv_heads * self.head_dim,
            bias=False
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        residual = x
        x = self.norm1(x)

        qkv = self.qkv_proj(x)
        q, k, v = torch.split(
            qkv,
            [self.embed_dim,
             self.num_kv_heads * self.head_dim,
             self.num_kv_heads * self.head_dim],
            dim=-1
        )

        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.num_query_groups, dim=1)
        v = v.repeat_interleave(self.num_query_groups, dim=1)

        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=self._get_mask(seq_len, x.device) if self.use_banded else None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=not self.use_banded
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if self.use_banded:
                scores = scores + self._get_mask(seq_len, x.device)
            scores = F.softmax(scores, dim=-1)
            scores = self.dropout(scores)
            attn_output = torch.matmul(scores, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i + 1] = 0
        return mask


class LightningOptimizedGQA_Transformer(pl.LightningModule):
    def __init__(
            self,
            vocab_size: int = 50257,
            embed_dim: int = 768,
            num_layers: int = 12,
            num_query_heads: int = 12,
            num_kv_heads: int = 3,
            window_size: int = 128,
            learning_rate: float = 3e-4,
            warmup_steps: int = 1000,
            max_steps: int = 100000,
            compile_model: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            LightningGQABlock(
                embed_dim=embed_dim,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                window_size=window_size,
                layer_idx=i
            )
            for i in range(num_layers)
        ])
        self.ln_f = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

        if compile_model and hasattr(torch, 'compile'):
            self.forward = torch.compile(self.forward, mode="reduce-overhead")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch['input_ids']
        labels = batch['labels']

        logits = self(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=torch.cuda.is_available()
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_steps - self.hparams.warmup_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


class JaxRotaryEmbedding(fnn.Module):
    dim: int
    max_seq_len: int = 131072
    base: float = 10000.0

    def setup(self):
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2) / self.dim))
        self.inv_freq = inv_freq

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, positions: jnp.ndarray) -> tuple:
        freqs = jnp.outer(positions, self.inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb)[None, :, None, :]
        sin = jnp.sin(emb)[None, :, None, :]

        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)

        return q_rot, k_rot

    def _rotate(self, x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = jnp.split(x, 2, axis=-1)
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        return jnp.concatenate([rx1, rx2], axis=-1)


class JaxGroupedQueryAttention(fnn.Module):
    embed_dim: int = 2880
    num_query_heads: int = 64
    num_kv_heads: int = 8
    window_size: int = 128
    use_banded: bool = False
    dropout_rate: float = 0.1

    def setup(self):
        self.head_dim = self.embed_dim // self.num_query_heads
        self.num_query_groups = self.num_query_heads // self.num_kv_heads
        self.scale = 1.0 / jnp.sqrt(self.head_dim)

        self.q_proj = fnn.Dense(self.embed_dim, use_bias=False)
        self.k_proj = fnn.Dense(self.num_kv_heads * self.head_dim, use_bias=False)
        self.v_proj = fnn.Dense(self.num_kv_heads * self.head_dim, use_bias=False)
        self.out_proj = fnn.Dense(self.embed_dim, use_bias=False)

        self.rope = JaxRotaryEmbedding(dim=self.head_dim)
        self.dropout = fnn.Dropout(self.dropout_rate)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        positions = jnp.arange(seq_len)
        q, k = self.rope(q, k, positions)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        k = jnp.repeat(k, self.num_query_groups, axis=1)
        v = jnp.repeat(v, self.num_query_groups, axis=1)

        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        if self.use_banded:
            mask = self._create_banded_mask(seq_len)
        else:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))

        attn_weights = jnp.where(mask[None, None, :, :], attn_weights, -1e10)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)

        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(attn_output)

    def _create_banded_mask(self, seq_len: int) -> jnp.ndarray:
        mask = jnp.zeros((seq_len, seq_len))
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask = mask.at[i, start:i + 1].set(1)
        return mask


class JaxTransformerBlock(fnn.Module):
    embed_dim: int
    num_query_heads: int
    num_kv_heads: int
    window_size: int
    layer_idx: int
    dropout_rate: float = 0.1

    def setup(self):
        self.use_banded = (self.layer_idx % 2 == 1)

        self.attention = JaxGroupedQueryAttention(
            embed_dim=self.embed_dim,
            num_query_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
            window_size=self.window_size,
            use_banded=self.use_banded,
            dropout_rate=self.dropout_rate
        )

        self.norm1 = fnn.LayerNorm()
        self.norm2 = fnn.LayerNorm()

        self.ffn = fnn.Sequential([
            fnn.Dense(4 * self.embed_dim),
            fnn.gelu,
            fnn.Dense(self.embed_dim),
            fnn.Dropout(self.dropout_rate)
        ])

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        residual = x
        x = self.norm1(x)
        x = self.attention(x, deterministic=deterministic)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x, deterministic=deterministic)
        x = residual + x

        return x


class JaxOptimizedGQA_Transformer(fnn.Module):
    vocab_size: int = 50257
    embed_dim: int = 768
    num_layers: int = 12
    num_query_heads: int = 12
    num_kv_heads: int = 3
    window_size: int = 128
    dropout_rate: float = 0.1

    def setup(self):
        self.token_embedding = fnn.Embed(self.vocab_size, self.embed_dim)

        self.blocks = [
            JaxTransformerBlock(
                embed_dim=self.embed_dim,
                num_query_heads=self.num_query_heads,
                num_kv_heads=self.num_kv_heads,
                window_size=self.window_size,
                layer_idx=i,
                dropout_rate=self.dropout_rate
            )
            for i in range(self.num_layers)
        ]

        self.ln_f = fnn.LayerNorm()
        self.lm_head = fnn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, input_ids: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x, deterministic=deterministic)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


def benchmark_pytorch_lightning():
    try:
        model = LightningOptimizedGQA_Transformer(
            vocab_size=10000,
            embed_dim=768,
            num_layers=12,
            num_query_heads=12,
            num_kv_heads=3,
            compile_model=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        batch_size = 4
        seq_len = 512
        input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)

        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()
        num_iterations = 10

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / num_iterations

        print(f"Device: {device}")
        print(f"Average forward pass time: {avg_time * 1000:.2f} ms")
        print(f"Throughput: {batch_size * seq_len / avg_time:.0f} tokens/sec")

        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e6:.1f}M")
        
    except Exception as e:
        print(f"Error in PyTorch benchmark: {str(e)}")
        import traceback
        traceback.print_exc()


def benchmark_jax():
    try:
        print("\n" + "=" * 60)
        print("JAX Performance")
        print("=" * 60)

        model = JaxOptimizedGQA_Transformer(
            vocab_size=10000,
            embed_dim=768,
            num_layers=12,
            num_query_heads=12,
            num_kv_heads=3
        )

        key = random.PRNGKey(0)
        batch_size = 4
        seq_len = 512
        dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        params = model.init(key, dummy_input)['params']

        @jit
        def forward(params, input_ids):
            return model.apply({'params': params}, input_ids, deterministic=True)

        input_ids = random.randint(key, (batch_size, seq_len), 0, 10000)

        for _ in range(3):
            _ = forward(params, input_ids).block_until_ready()

        start_time = time.time()
        num_iterations = 10

        for _ in range(num_iterations):
            _ = forward(params, input_ids).block_until_ready()

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / num_iterations

        print(f"Device: {jax.devices()[0]}")
        print(f"Average forward pass time: {avg_time * 1000:.2f} ms")
        print(f"Throughput: {batch_size * seq_len / avg_time:.0f} tokens/sec")

        total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Total parameters: {total_params / 1e6:.1f}M")

        print("\n" + "-" * 40)
        print("Advanced JAX Features:")
        print("-" * 40)

        def loss_fn(params, input_ids):
            logits = model.apply({'params': params}, input_ids, deterministic=True)
            return jnp.mean(logits)

        grad_fn = jit(jax.grad(loss_fn))
        grads = grad_fn(params, input_ids)
        
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        
        print("Automatic differentiation (grad): computed gradients")
        print(f"Gradient norm: {grad_norm:.4f}")
        
        learning_rate = 1e-4
        updated_params = jax.tree_util.tree_map(
            lambda p, g: p - learning_rate * g, 
            params, 
            grads
        )
        
        print("Performed parameter update step with SGD")
        
        param_diff = jax.tree_util.tree_map(
            lambda a, b: jnp.sum((a - b)**2), 
            params, 
            updated_params
        )
        total_diff = sum(jax.tree_util.tree_leaves(param_diff))
        print(f"Total parameter change: {total_diff:.8f}")
        
    except Exception as e:
        print(f"Error in JAX benchmark: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        print("Starting benchmarks...")
        benchmark_pytorch_lightning()
        benchmark_jax()
        print("\n" + "=" * 60)
        print("All benchmarks completed successfully!")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\nBenchmarking interrupted by user.")
    except Exception as e:
        print(f"\nBenchmarking failed with error: {str(e)}")
        import traceback
        traceback.print_exc()