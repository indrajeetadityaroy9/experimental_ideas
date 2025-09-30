import torch
import numpy as np

def sym_norm(adjacency_matrix):
    degree = torch.sum(adjacency_matrix, dim=1).pow(-0.5)
    degree[torch.isinf(degree)] = 0.
    return degree.unsqueeze(1) * adjacency_matrix * degree.unsqueeze(0)

def row_topk(adjacency_matrix, k):
    if adjacency_matrix.size(0) <= k:
        result = adjacency_matrix.clone()
    else:
        sparse_matrix = torch.zeros_like(adjacency_matrix)
        values, indices = torch.topk(adjacency_matrix, k=k, dim=1)
        sparse_matrix.scatter_(1, indices, values)
        result = torch.maximum(sparse_matrix, sparse_matrix.t())

    result = result.clone()
    result.fill_diagonal_(0)
    return result

def readout_features(x):
    if x.numel() == 0:
        raise ValueError('Cannot readout from empty feature matrix')
    mean = x.mean(dim=0)
    max_vals = x.max(dim=0).values
    return torch.cat([mean, max_vals], dim=0)

def board_node_features(board_np):
    height, width = board_np.shape
    black = (board_np == 1).astype(np.float32).reshape(-1, 1)
    white = (board_np == 2).astype(np.float32).reshape(-1, 1)
    empty = (board_np == 0).astype(np.float32).reshape(-1, 1)

    rows = np.arange(height).repeat(width).astype(np.float32) / max(1., height - 1)
    cols = np.tile(np.arange(width), height).astype(np.float32) / max(1., width - 1)
    rc = np.stack([rows, cols], axis=1)

    stacked = np.concatenate([black, white, empty, rc], axis=1)
    return torch.from_numpy(stacked)

def lattice_edge_index(height, width):
    edges = []

    def node_id(row, col):
        return row * width + col

    for row in range(height):
        for col in range(width):
            node = node_id(row, col)

            if row + 1 < height:
                edges.extend([(node, node_id(row + 1, col)), (node_id(row + 1, col), node)])
            if col + 1 < width:
                edges.extend([(node, node_id(row, col + 1)), (node_id(row, col + 1), node)])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()
