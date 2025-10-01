# GQA Comparison
Benchmark utilities for Grouped-Query Attention (GQA) blocks implemented in **PyTorch Lightning** and **JAX/Flax**. The CLI measures forward latency, tokens/sec, parameter counts, and gradient diagnostics across both backends.

## Grouped-Query Attention
In grouped-query attention, query heads are bucketed into groups that share key/value projections.
```math
\begin{aligned}
\mathrm{Queries:} & \;\; \mathrm{Q} \in \mathbb{R}^{B \times H_q \times L \times d} \\
\mathrm{Keys/Values:} & \;\; \mathrm{K}, \mathrm{V} \in \mathbb{R}^{B \times H_k \times L \times d} \\
\mathrm{Grouping\;factor:} & \;\; g = \tfrac{H_q}{H_k}
\end{aligned}
```

For each group \( i \):
```math
\mathrm{Attn}_i(Q, K, V) = \mathrm{softmax}\!\left(\tfrac{Q_i K_i^\top}{\sqrt{d}} + M \right)\, V_i
```
where:
* \( Q_i \) are the \( g \) query heads mapped to key/value head \( i \)  
* \( M \) encodes the attention mask (causal for even layers, banded for odd layers)

```math
\mathrm{GQA}(Q, K, V) = \mathrm{Proj}\!\left(\mathrm{concat}_i \, \mathrm{Attn}_i(Q, K, V)\right)
```
