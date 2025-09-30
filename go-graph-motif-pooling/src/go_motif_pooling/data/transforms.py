import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from go_motif_pooling.game.patterns import MotifConv2D, motif_adjacency_from_hits
from go_motif_pooling.utils.graph import lattice_edge_index, board_node_features

def board_to_data(board_np, target=None, motif_conv=None):
    if motif_conv is None:
        motif_conv = MotifConv2D()

    height, width = board_np.shape

    edge_index = lattice_edge_index(height, width)

    node_features = board_node_features(board_np).float()

    hits = motif_conv(board_np)
    motif_adjacency = motif_adjacency_from_hits(hits, height, width)
    motif_edge_index, motif_edge_weight = dense_to_sparse(motif_adjacency)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        motif_edge_index=motif_edge_index,
        motif_edge_weight=motif_edge_weight
    )

    data.num_nodes = height * width
    board_tensor = torch.from_numpy(board_np.astype(np.int64))
    data.board_state = board_tensor.view(-1)
    data.board_size = torch.tensor([height, width], dtype=torch.long)

    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float32)

    return data

def boards_to_dataset(boards, targets=None, motif_conv=None):
    if motif_conv is None:
        motif_conv = MotifConv2D()

    if targets is None:
        targets = [None] * len(boards)

    graphs = []
    for board, target in zip(boards, targets):
        data = board_to_data(board, target=target, motif_conv=motif_conv)
        graphs.append(data)

    return graphs
