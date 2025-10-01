# GQA Comparison
Benchmark utilities for grouped-query attention (GQA) blocks implemented in PyTorch Lightning and JAX/Flax. The CLI measures forward latency, tokens/sec, parameter counts, and gradient diagnostics across both backends.

## Grouped-Query Attention
In grouped-query attention, query heads are bucketed into groups that share key/value projections. Given $Q \in \mathbb{R}^{B \times H_q \times L \times d}$ and shared $K, V \in \mathbb{R}^{B \times H_k \times L \times d}$ with grouping factor $g = H_q / H_k$, the attention weights for group $i$ are

$$
\operatorname{Attn}_i(Q, K, V) = \operatorname{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d}} + M\right) V_i,
$$

where $Q_i$ are the $g$ query heads mapped to key/value head $i$ and $M$ encodes the attention mask (causal for even layers, banded for odd layers). The final output concatenates per-group results and applies the output projection: $\mathrm{GQA}(Q, K, V) = \mathrm{Proj}(\operatorname{concat}_i \operatorname{Attn}_i(Q, K, V))$.

