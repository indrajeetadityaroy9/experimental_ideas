import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class MotifSelectionPool(nn.Module):
    def __init__(self, in_channels, pool_ratio):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.score_gnn = GCNConv(in_channels, in_channels, add_self_loops=False)
        self.score_linear = nn.Linear(in_channels, 1)

    def forward(self, x, adjacency, motif_adjacency):
        num_nodes = x.size(0)
        if num_nodes == 0:
            raise ValueError('Selection pool received empty graph')

        motif_norm = motif_adjacency + torch.eye(num_nodes, device=x.device)
        edge_index, edge_weight = dense_to_sparse(motif_norm)

        h = torch.tanh(self.score_gnn(x, edge_index, edge_weight=edge_weight))
        scores = self.score_linear(h).squeeze(-1)

        k = max(1, min(num_nodes, int(torch.ceil(torch.tensor(self.pool_ratio * num_nodes)).item())))
        _, top_idx = torch.topk(scores, k=k)
        top_idx, _ = torch.sort(top_idx)

        x_pool = h[top_idx]
        adjacency_pool = adjacency[top_idx][:, top_idx]
        motif_pool = motif_adjacency[top_idx][:, top_idx]

        return {
            'x': x_pool,
            'adjacency': adjacency_pool,
            'motif_adjacency': motif_pool,
            'scores': scores.detach(),
            'indices': top_idx.detach()
        }

class MotifClusteringPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, clusters):
        super().__init__()
        self.gnn = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        self.assignment_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, clusters)
        )
        self.clusters = clusters

    def forward(self, x, adjacency, motif_adjacency):
        num_nodes = x.size(0)
        if num_nodes == 0:
            raise ValueError('Clustering pool received empty graph')

        motif_norm = motif_adjacency + torch.eye(num_nodes, device=x.device)
        edge_index, edge_weight = dense_to_sparse(motif_norm)

        h = torch.tanh(self.gnn(x, edge_index, edge_weight=edge_weight))
        logits = self.assignment_mlp(h)
        S = torch.softmax(logits, dim=1)

        X_pool = S.t() @ h
        A_pool = S.t() @ adjacency @ S
        A_pool.fill_diagonal_(0)
        motif_pool = S.t() @ motif_adjacency @ S
        motif_pool.fill_diagonal_(0)

        degree = motif_adjacency.sum(dim=1)
        D = torch.diag(degree)
        cut_num = torch.trace(S.t() @ motif_adjacency @ S)
        cut_den = torch.trace(S.t() @ D @ S) + 1e-8
        L_cut = -cut_num / cut_den

        StS = S.t() @ S
        eye = torch.eye(self.clusters, device=S.device)
        L_ortho = torch.norm(StS - eye, p='fro') / (self.clusters ** 0.5)

        losses = {
            'cut': L_cut,
            'ortho': L_ortho
        }

        return {
            'x': X_pool,
            'adjacency': A_pool,
            'motif_adjacency': motif_pool,
            'S': S.detach(),
            'losses': losses
        }
