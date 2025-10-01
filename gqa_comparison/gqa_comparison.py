import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

import jax
import jax.numpy as jnp
from jax import jit, random
import flax.linen as fnn


def torch_is_compiling():
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return compiler.is_compiling()
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return dynamo.is_compiling()
    return False


class BenchmarkConfig:
    def __init__(
        self,
        backend,
        vocab_size=10000,
        embed_dim=768,
        num_layers=12,
        num_query_heads=12,
        num_kv_heads=3,
        window_size=128,
        batch_size=4,
        seq_len=512,
        warmup_iters=3,
        timed_iters=10,
        learning_rate=3e-4,
        compile_model=True,
        show_grad_demo=False,
    ):
        self.backend = backend
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters
        self.learning_rate = learning_rate
        self.compile_model = compile_model
        self.show_grad_demo = show_grad_demo

        if self.num_kv_heads == 0 or self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads and num_kv_heads must be non-zero"
            )


class BenchmarkResult:
    def __init__(self, backend, device, avg_forward_ms, tokens_per_second, total_params_m, extra=None):
        self.backend = backend
        self.device = device
        self.avg_forward_ms = avg_forward_ms
        self.tokens_per_second = tokens_per_second
        self.total_params_m = total_params_m
        self.extra = extra or {}

    def pretty(self):
        header = f"[{self.backend}] device={self.device}"
        metrics = [
            f"avg={self.avg_forward_ms:.2f} ms",
            f"throughput={self.tokens_per_second:.0f} tok/s",
            f"params={self.total_params_m:.1f}M",
        ]
        metrics.extend(f"{k}={v:.4f}" for k, v in self.extra.items())
        return header + " | " + ", ".join(metrics)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


class LightningGQABlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_query_heads,
        num_kv_heads,
        window_size,
        layer_idx,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_query_groups = num_query_heads // num_kv_heads
        self.use_banded = (layer_idx % 2) == 1
        self.window_size = window_size

        self.qkv_proj = nn.Linear(
            embed_dim,
            embed_dim + 2 * self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self._mask_cache = {}

    def _build_banded_mask(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        delta = positions[:, None] - positions[None, :]
        allowed = (delta >= 0) & (delta < self.window_size)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return mask.masked_fill(allowed, 0.0)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(
            qkv,
            [
                self.embed_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        batch_size, seq_len = x.size(0), x.size(1)
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_query_groups > 1:
            if k.is_cuda:
                k = k.view(batch_size, self.num_kv_heads, 1, seq_len, self.head_dim)
                k = k.expand(-1, -1, self.num_query_groups, -1, -1)
                k = k.reshape(batch_size, self.num_query_heads, seq_len, self.head_dim).contiguous()

                v = v.view(batch_size, self.num_kv_heads, 1, seq_len, self.head_dim)
                v = v.expand(-1, -1, self.num_query_groups, -1, -1)
                v = v.reshape(batch_size, self.num_query_heads, seq_len, self.head_dim).contiguous()
            else:
                k = k.repeat_interleave(self.num_query_groups, dim=1)
                v = v.repeat_interleave(self.num_query_groups, dim=1)

        attn_mask = None
        if self.use_banded:
            cache_key = (seq_len, x.device)
            if torch_is_compiling():
                attn_mask = self._build_banded_mask(seq_len, x.device)
            else:
                attn_mask = self._mask_cache.get(cache_key)
                if attn_mask is None:
                    attn_mask = self._build_banded_mask(seq_len, x.device)
                    self._mask_cache[cache_key] = attn_mask

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=not self.use_banded,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        x = residual + self.dropout(self.out_proj(attn_output))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class LightningOptimizedGQA_Transformer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(dict(vars(cfg)))
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.blocks = nn.ModuleList(
            LightningGQABlock(
                embed_dim=cfg.embed_dim,
                num_query_heads=cfg.num_query_heads,
                num_kv_heads=cfg.num_kv_heads,
                window_size=cfg.window_size,
                layer_idx=i,
            )
            for i in range(cfg.num_layers)
        )
        self.ln_f = RMSNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        if cfg.compile_model and hasattr(torch, "compile"):
            self.forward = torch.compile(self.forward, mode="reduce-overhead")

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=-100,
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=torch.cuda.is_available(),
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams["timed_iters"],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def benchmark_pytorch(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightningOptimizedGQA_Transformer(cfg).to(device).eval()

    input_ids = torch.randint(
        0,
        cfg.vocab_size,
        (cfg.batch_size, cfg.seq_len),
        device=device,
    )

    with torch.no_grad():
        for _ in range(cfg.warmup_iters):
            model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(cfg.timed_iters):
            model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

    extra = {}
    if cfg.show_grad_demo:
        dummy_target = torch.randint(0, cfg.vocab_size, input_ids.shape, device=device)
        loss = F.cross_entropy(model(input_ids).flatten(0, 1), dummy_target.flatten(), ignore_index=-100)
        loss.backward()
        grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None))
        extra["grad_norm"] = float(grad_norm.detach().cpu())
        model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        extra["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3

    avg_time = elapsed / cfg.timed_iters
    total_params = sum(p.numel() for p in model.parameters())
    return BenchmarkResult(
        backend="pytorch",
        device=str(device),
        avg_forward_ms=avg_time * 1_000,
        tokens_per_second=(cfg.batch_size * cfg.seq_len) / avg_time,
        total_params_m=total_params / 1e6,
        extra=extra,
    )
class JaxRotaryEmbedding(fnn.Module):
    dim: int
    max_seq_len: int = 131072
    base: float = 10000.0

    def setup(self):
        self.inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2) / self.dim))

    def __call__(self, q, k, positions):
        freqs = jnp.outer(positions, self.inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos, sin = jnp.cos(emb)[None, :, None, :], jnp.sin(emb)[None, :, None, :]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(self, x, cos, sin):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        cos_even = cos[..., ::2]
        sin_even = sin[..., ::2]
        rotated_even = x_even * cos_even - x_odd * sin_even
        rotated_odd = x_even * sin_even + x_odd * cos_even
        return jnp.stack([rotated_even, rotated_odd], axis=-1).reshape(x.shape)


class JaxGroupedQueryAttention(fnn.Module):
    embed_dim: int
    num_query_heads: int
    num_kv_heads: int
    window_size: int
    use_banded: bool
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
        self.attn_dropout = fnn.Dropout(self.dropout_rate)
        self.out_dropout = fnn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        positions = jnp.arange(seq_len)
        q, k = self.rope(q, k, positions)
        q, k, v = [t.transpose(0, 2, 1, 3) for t in (q, k, v)]
        if self.num_query_groups > 1:
            k = jnp.reshape(k, (batch_size, self.num_kv_heads, 1, seq_len, self.head_dim))
            k = jnp.broadcast_to(
                k,
                (batch_size, self.num_kv_heads, self.num_query_groups, seq_len, self.head_dim),
            )
            k = jnp.reshape(k, (batch_size, self.num_query_heads, seq_len, self.head_dim))

            v = jnp.reshape(v, (batch_size, self.num_kv_heads, 1, seq_len, self.head_dim))
            v = jnp.broadcast_to(
                v,
                (batch_size, self.num_kv_heads, self.num_query_groups, seq_len, self.head_dim),
            )
            v = jnp.reshape(v, (batch_size, self.num_query_heads, seq_len, self.head_dim))

        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * self.scale
        delta = positions[:, None] - positions[None, :]
        if self.use_banded:
            allowed = (delta >= 0) & (delta < self.window_size)
        else:
            allowed = delta >= 0
        attn_weights = jnp.where(allowed[None, None, :, :], attn_weights, -1e9)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, deterministic=deterministic)
        out = jnp.matmul(attn_weights, v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return self.out_dropout(out, deterministic=deterministic)


class JaxTransformerBlock(fnn.Module):
    embed_dim: int
    num_query_heads: int
    num_kv_heads: int
    window_size: int
    layer_idx: int
    dropout_rate: float = 0.1

    def setup(self):
        self.use_banded = (self.layer_idx % 2) == 1
        self.attention = JaxGroupedQueryAttention(
            embed_dim=self.embed_dim,
            num_query_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
            window_size=self.window_size,
            use_banded=self.use_banded,
            dropout_rate=self.dropout_rate,
        )
        self.norm1 = fnn.LayerNorm()
        self.norm2 = fnn.LayerNorm()
        self.ffn_dense1 = fnn.Dense(4 * self.embed_dim)
        self.ffn_dense2 = fnn.Dense(self.embed_dim)
        self.ffn_dropout = fnn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic):
        residual = x
        x = self.norm1(x)
        attn_out = self.attention(x, deterministic)
        x = residual + attn_out
        ffn_input = self.norm2(x)
        ffn_hidden = self.ffn_dense1(ffn_input)
        ffn_hidden = fnn.gelu(ffn_hidden)
        ffn_hidden = self.ffn_dense2(ffn_hidden)
        ffn_hidden = self.ffn_dropout(ffn_hidden, deterministic=deterministic)
        x = x + ffn_hidden
        return x


class JaxOptimizedGQA_Transformer(fnn.Module):
    vocab_size: int
    embed_dim: int
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    window_size: int
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
                dropout_rate=self.dropout_rate,
            )
            for i in range(self.num_layers)
        ]
        self.ln_f = fnn.LayerNorm()
        self.lm_head = fnn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, input_ids, deterministic=True):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        return self.lm_head(self.ln_f(x))


def benchmark_jax(cfg):
    model = JaxOptimizedGQA_Transformer(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        window_size=cfg.window_size,
    )

    key = random.PRNGKey(0)
    dummy_input = jnp.ones((cfg.batch_size, cfg.seq_len), dtype=jnp.int32)
    params = model.init(key, dummy_input)["params"]
    forward = jit(lambda p, x: model.apply({"params": p}, x, deterministic=True))

    input_ids = random.randint(key, dummy_input.shape, 0, cfg.vocab_size)

    for _ in range(cfg.warmup_iters):
        forward(params, input_ids).block_until_ready()

    start_time = time.perf_counter()
    for _ in range(cfg.timed_iters):
        forward(params, input_ids).block_until_ready()
    elapsed = time.perf_counter() - start_time

    extra = {}
    if cfg.show_grad_demo:
        def loss_fn(p, ids):
            logits = model.apply({"params": p}, ids, deterministic=True)
            return jnp.mean(logits)

        grad_fn = jit(jax.grad(loss_fn))
        grads = grad_fn(params, input_ids)
        grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        extra["grad_norm"] = float(grad_norm)

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    avg_time = elapsed / cfg.timed_iters
    return BenchmarkResult(
        backend="jax",
        device=str(jax.devices()[0]),
        avg_forward_ms=avg_time * 1_000,
        tokens_per_second=(cfg.batch_size * cfg.seq_len) / avg_time,
        total_params_m=total_params / 1e6,
        extra=extra,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare grouped-query attention backends.")
    parser.add_argument("--backend", choices=["pytorch", "jax", "both"], default="both")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile().")
    parser.add_argument("--grad-demo", action="store_true", help="Measure gradient norm after forward.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        backend=args.backend,
        compile_model=not args.no_compile,
        show_grad_demo=args.grad_demo,
        warmup_iters=args.warmup,
        timed_iters=args.iters,
    )

    results = []
    if args.backend in ("pytorch", "both"):
        results.append(benchmark_pytorch(cfg))
    if args.backend in ("jax", "both"):
        results.append(benchmark_jax(cfg))

    print("=" * 80)
    for result in results:
        print(result.pretty())
    print("=" * 80)


if __name__ == "__main__":
    main()
