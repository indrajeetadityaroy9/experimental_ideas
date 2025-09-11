import sys
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse


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


def board_one_hot(board_np):
    height, width = board_np.shape
    black = torch.tensor((board_np == 1).astype(np.float32)).view(1, 1, height, width)
    white = torch.tensor((board_np == 2).astype(np.float32)).view(1, 1, height, width)
    empty = torch.tensor((board_np == 0).astype(np.float32)).view(1, 1, height, width)
    return black, white, empty


def _conv_hits(x, kernel):
    return F.conv2d(x, kernel, stride=1, padding=0)


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


def motif_adjacency_from_hits(hits, height, width):
    num_nodes = height * width
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    if not hits:
        return adjacency

    def node_id(row, col):
        return row * width + col

    def add_2x2(row, col, weight):
        cells = [(row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1)]
        indices = [node_id(x, y) for x, y in cells if 0 <= x < height and 0 <= y < width]
        for u in indices:
            for v in indices:
                if u != v:
                    adjacency[u, v] += weight

    def add_eye3x3(row, col, weight):
        center = (row + 1, col + 1)
        ring = [(center[0] - 1, center[1]), (center[0] + 1, center[1]),
                (center[0], center[1] - 1), (center[0], center[1] + 1)]
        indices = [node_id(x, y) for x, y in ring if 0 <= x < height and 0 <= y < width]
        for u in indices:
            for v in indices:
                if u != v:
                    adjacency[u, v] += weight

    weights = {'bamboo': 1.2, 'tiger': 1.0, 'empty_triangle': 0.8}

    for pattern, weight in weights.items():
        for color in ['black', 'white']:
            pattern_mask = hits[f'{pattern}_{color}']
            rows, cols = torch.where(pattern_mask)
            for row, col in zip(rows.tolist(), cols.tolist()):
                add_2x2(row, col, weight)

    for color in ['black', 'white']:
        eye_mask = hits[f'eye_{color}']
        rows, cols = torch.where(eye_mask)
        for row, col in zip(rows.tolist(), cols.tolist()):
            add_eye3x3(row, col, 1.5)

    return torch.maximum(adjacency, adjacency.t())


class MotifConv2D(nn.Module):

    def __init__(self):
        super().__init__()
        diag = torch.tensor([[1., 0.], [0., 1.]]).view(1, 1, 2, 2)
        self.register_buffer('k_diag', diag)
        anti = torch.tensor([[0., 1.], [1., 0.]]).view(1, 1, 2, 2)
        self.register_buffer('k_anti', anti)
        all_2 = torch.ones(1, 1, 2, 2)
        self.register_buffer('k_all2', all_2)
        cross = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('k_cross', cross)
        center = torch.zeros_like(cross)
        center[0, 0, 1, 1] = 1.
        self.register_buffer('k_center', center)
        d_e = torch.tensor([[0, 0], [0, 1]]).view(1, 1, 2, 2).float()
        a_e = torch.tensor([[1, 0], [0, 0]]).view(1, 1, 2, 2).float()
        b_e = torch.tensor([[0, 1], [0, 0]]).view(1, 1, 2, 2).float()
        c_e = torch.tensor([[0, 0], [1, 0]]).view(1, 1, 2, 2).float()
        self.register_buffer('tmp_de', d_e)
        self.register_buffer('tmp_ae', a_e)
        self.register_buffer('tmp_be', b_e)
        self.register_buffer('tmp_ce', c_e)

    @torch.no_grad()
    def forward(self, board_np):
        height, width = board_np.shape
        if height < 3 or width < 3:
            return {}

        black, white, empty = board_one_hot(board_np)

        def squeeze_bool(x):
            return x.to(torch.bool).squeeze(0).squeeze(0)

        hits = {}
        black_diag = _conv_hits(black, self.k_diag)
        empty_anti = _conv_hits(empty, self.k_anti)
        black_anti = _conv_hits(black, self.k_anti)
        empty_diag = _conv_hits(empty, self.k_diag)
        hits['bamboo_black'] = squeeze_bool(
            ((black_diag == 2) & (empty_anti == 2)) | ((black_anti == 2) & (empty_diag == 2))
        )

        white_diag = _conv_hits(white, self.k_diag)
        white_anti = _conv_hits(white, self.k_anti)
        hits['bamboo_white'] = squeeze_bool(
            ((white_diag == 2) & (empty_anti == 2)) | ((white_anti == 2) & (empty_diag == 2))
        )

        black_s2 = _conv_hits(black, self.k_all2)
        empty_s2 = _conv_hits(empty, self.k_all2)
        empty_d = _conv_hits(empty, self.tmp_de)
        empty_a = _conv_hits(empty, self.tmp_ae)
        empty_b = _conv_hits(empty, self.tmp_be)
        empty_c = _conv_hits(empty, self.tmp_ce)
        hits['tiger_black'] = squeeze_bool(
            (black_s2 >= 2) & ((empty_d == 1) | (empty_a == 1) | (empty_b == 1) | (empty_c == 1))
        )

        white_s2 = _conv_hits(white, self.k_all2)
        hits['tiger_white'] = squeeze_bool(
            (white_s2 >= 2) & ((empty_d == 1) | (empty_a == 1) | (empty_b == 1) | (empty_c == 1))
        )

        hits['empty_triangle_black'] = squeeze_bool((black_s2 == 3) & (empty_s2 == 1))
        hits['empty_triangle_white'] = squeeze_bool((white_s2 == 3) & (empty_s2 == 1))

        empty_center = _conv_hits(empty, self.k_center)
        black_cross = _conv_hits(black, self.k_cross)
        white_cross = _conv_hits(white, self.k_cross)
        hits['eye_black'] = squeeze_bool((empty_center == 1) & (black_cross == 4))
        hits['eye_white'] = squeeze_bool((empty_center == 1) & (white_cross == 4))

        return hits


class DSRBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, pool_clusters, radius=2, topk_add=2,
                 motif_topk=8, eta=0.5, l0_gamma=-0.1, l0_zeta=1.1, initial_tau=1.0,
                 tau_anneal_factor=0.95, **loss_weights):
        super().__init__()
        self.pool_clusters = pool_clusters
        self.radius = radius
        self.topk_add = topk_add
        self.motif_topk = motif_topk
        self.eta = eta
        self.gamma = l0_gamma
        self.zeta = l0_zeta
        self.tau = nn.Parameter(torch.tensor(initial_tau))
        self.tau_anneal_factor = tau_anneal_factor
        self.lambda_map = loss_weights

        self.gat_A = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.gcn_M = GCNConv(in_channels, hidden_channels * 2, add_self_loops=False)
        self.bn_A = nn.BatchNorm1d(hidden_channels * 2)
        self.bn_M = nn.BatchNorm1d(hidden_channels * 2)
        self.mu = nn.Parameter(torch.tensor(-3.0))
        self.prune_mlp = nn.Linear(hidden_channels * 4 + 1, 1)
        self.rewire_mlp = nn.Linear(hidden_channels * 4 + 1, 1)
        self.pool_gnn = GCNConv(hidden_channels * 2, pool_clusters)
        self.gamma_pool = nn.Parameter(torch.tensor(-2.0))
        self._cached_candidates = {}

    def _get_rewire_candidates(self, adj, coords, device):
        distance = torch.cdist(coords, coords, p=1)
        mask = (distance > 0) & (distance <= self.radius) & (adj < 1e-6)
        triu_indices = torch.triu_indices(adj.size(0), adj.size(0), offset=1, device=device)
        candidate_indices = triu_indices[:, mask[triu_indices[0], triu_indices[1]]]
        return candidate_indices

    def _hard_concrete_gate(self, logits):
        tau = self.tau.clamp(min=0.1)
        if self.training:
            uniform = torch.rand_like(logits)
            sigmoid = torch.sigmoid((torch.log(uniform) - torch.log(1 - uniform) + logits) / tau)
            s_bar = sigmoid * (self.zeta - self.gamma) + self.gamma
            z = F.hardtanh(s_bar, min_val=0, max_val=1)
        else:
            sigmoid = torch.sigmoid(logits / tau)
            z = F.hardtanh(sigmoid * (self.zeta - self.gamma) + self.gamma, min_val=0, max_val=1)
        return z

    def forward(self, x, A_in, A_motif, coords):
        num_nodes = x.size(0)
        edge_index_A, _ = dense_to_sparse(A_in)
        h_A = F.elu(self.bn_A(self.gat_A(x, edge_index_A)))
        M_hat = row_topk(A_motif, k=self.motif_topk)
        edge_index_M, edge_weight_M = dense_to_sparse(M_hat)
        h_M = F.elu(self.bn_M(self.gcn_M(x, edge_index_M, edge_weight=edge_weight_M)))
        h = h_A + F.softplus(self.mu) * h_M
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=x.device)
        mask = A_in[triu_indices[0], triu_indices[1]] > 0
        und_row, und_col = triu_indices[:, mask]

        prune_features = torch.cat([
            h[und_row],
            h[und_col],
            M_hat[und_row, und_col].unsqueeze(1)
        ], dim=-1)
        prune_logits = self.prune_mlp(prune_features).squeeze()
        z_prune = self._hard_concrete_gate(prune_logits)
        A_pruned = torch.zeros_like(A_in)
        A_pruned[und_row, und_col] = z_prune * A_in[und_row, und_col]
        A_pruned[und_col, und_row] = z_prune * A_in[und_col, und_row]

        candidate_indices = self._get_rewire_candidates(A_in, coords, x.device)
        if candidate_indices.numel() > 0:
            row_a, col_a = candidate_indices
            add_features = torch.cat([
                h[row_a],
                h[col_a],
                M_hat[row_a, col_a].unsqueeze(1)
            ], dim=-1)
            rewire_logits = self.rewire_mlp(add_features).squeeze()
            z_add = self._hard_concrete_gate(rewire_logits)

            with torch.no_grad():
                add_mask = torch.zeros_like(z_add, dtype=torch.bool)
                for i in range(num_nodes):
                    is_node_i = (row_a == i) | (col_a == i)
                    num_candidates = is_node_i.sum().item()
                    if num_candidates > 0 and num_candidates <= self.topk_add:
                        add_mask[is_node_i] = True
                    elif num_candidates > self.topk_add:
                        scores_i = z_add[is_node_i]
                        k = min(self.topk_add, num_candidates)
                        _, topk_idx = torch.topk(scores_i, k)
                        orig_idx = torch.where(is_node_i)[0]
                        add_mask[orig_idx[topk_idx]] = True
                z_add = z_add * add_mask
        else:
            rewire_logits = torch.empty(0, device=x.device)
            z_add = torch.empty(0, device=x.device)

        A_added = to_dense_adj(
            candidate_indices,
            edge_attr=z_add,
            max_num_nodes=num_nodes
        ).squeeze(0) if candidate_indices.numel() > 0 else torch.zeros_like(A_pruned)

        A_added_sym = torch.maximum(A_added, A_added.t())
        A_refined = A_pruned + self.eta * A_added_sym

        ref_idx, ref_w = dense_to_sparse(A_refined)
        S = torch.softmax(self.pool_gnn(h, ref_idx, ref_w), dim=1)
        X_coarse = S.t() @ h
        A_coarse_dense = S.t() @ A_refined @ S
        A_coarse_dense.fill_diagonal_(0)
        A_coarse = row_topk(A_coarse_dense, k=self.motif_topk)

        losses, gamma_value = self._compute_losses(S, A_refined, M_hat, prune_logits, rewire_logits)
        new_coords = (S.t() @ coords) / (S.sum(dim=0, keepdim=True).t() + 1e-8)

        kept_edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        if z_prune.numel() > 0:
            kept_mask = z_prune > 0.1
            if kept_mask.any():
                kept_edge_index = torch.stack([und_row[kept_mask], und_col[kept_mask]], dim=0)
                kept_edge_index = torch.cat([
                    kept_edge_index,
                    torch.stack([und_col[kept_mask], und_row[kept_mask]], dim=0)
                ], dim=1)

        added_edge_index = (
            candidate_indices[:, z_add > 0.1].detach()
            if candidate_indices.numel() > 0
            else torch.empty((2, 0), dtype=torch.long, device=x.device)
        )

        diagnostics = {
            'losses': {k: v.detach().cpu().item() for k, v in losses.items()},
            'S': S.detach(),
            'kept_edge_index': kept_edge_index,
            'added_edge_index': added_edge_index,
            'A_coarse': A_coarse.detach(),
            'X_coarse': X_coarse,
            'coords': new_coords.detach(),
            'input_coords': coords.detach(),
            'gamma': gamma_value
        }

        return X_coarse, A_coarse, diagnostics, losses['total']

    def _compute_losses(self, S, A_refined, M_hat, prune_logits, rewire_logits):
        gamma = torch.sigmoid(self.gamma_pool)
        gamma_value = gamma.detach().cpu().item()

        A_hat_refined = sym_norm(A_refined)
        M_hat_loss = sym_norm(M_hat)

        A_cut_obj = (1 - gamma) * A_hat_refined + gamma * M_hat_loss
        degree = torch.sum(A_cut_obj, dim=1)
        D_cut = torch.diag(degree)
        L_cut_mat = D_cut - A_cut_obj
        L_cut = torch.trace(S.t() @ L_cut_mat @ S) / (torch.trace(S.t() @ D_cut @ S) + 1e-8)

        StDS = S.t() @ D_cut @ S
        identity = torch.eye(self.pool_clusters, device=S.device)
        L_ortho = torch.norm(StDS / (torch.trace(D_cut) + 1e-8) - identity / self.pool_clusters, p='fro') ** 2
        L_bal = -torch.sum(S.mean(dim=0) * torch.log(S.mean(dim=0).clamp(min=1e-8)))

        l0_shift = torch.log(torch.tensor(-self.gamma / self.zeta, device=S.device))
        L_L0 = torch.sigmoid(prune_logits - l0_shift).mean() + (
            torch.sigmoid(rewire_logits - l0_shift).mean()
            if rewire_logits.numel() > 0
            else 0.
        )

        L_motif = -torch.sum(A_hat_refined * M_hat_loss) / (
                torch.norm(A_hat_refined) * torch.norm(M_hat_loss) + 1e-8
        )

        loss_map = self.lambda_map
        total = (
                loss_map['lambda_cut'] * L_cut +
                loss_map['lambda_motif'] * L_motif +
                loss_map['lambda_L0'] * L_L0 +
                loss_map['lambda_ortho'] * L_ortho +
                loss_map['lambda_bal'] * L_bal
        )

        return {
            'total': total,
            'cut': L_cut,
            'motif': L_motif,
            'L0': L_L0,
            'ortho': L_ortho,
            'bal': L_bal
        }, gamma_value


class GoPyG_DSR_Model(nn.Module):
    def __init__(self, size=9, in_dim=5, hidden=32, num_layers=2, pool_ratios=[0.25, 0.25]):
        super().__init__()
        self.mconv = MotifConv2D()
        self.initial_feature_map = nn.Linear(in_dim, hidden)
        self.dsr_blocks = nn.ModuleList()

        current_channels = hidden
        num_nodes = size * size

        for i in range(num_layers):
            pool_clusters = max(2, int(num_nodes * pool_ratios[i]))
            motif_topk = 12 if i == 0 else 8
            block = DSRBlock(
                current_channels,
                hidden,
                pool_clusters,
                radius=2,
                topk_add=2,
                motif_topk=motif_topk,
                tau_anneal_factor=0.95,
                lambda_cut=1.0,
                lambda_motif=0.15,
                lambda_L0=1e-3,
                lambda_ortho=1.0,
                lambda_bal=0.2
            )
            self.dsr_blocks.append(block)
            current_channels = hidden * 2
            num_nodes = pool_clusters

        self.final_mlp = nn.Sequential(
            nn.Linear(current_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, board_np):
        height, width = board_np.shape
        device = self.initial_feature_map.weight.device
        edge_index = lattice_edge_index(height, width).to(device)
        x = self._node_features(board_np).to(device)
        A_motif = motif_adjacency_from_hits(self.mconv(board_np), height, width).to(device)
        x = self.initial_feature_map(x)

        diagnostics = []
        total_reg_loss = 0
        A_current = to_dense_adj(edge_index, max_num_nodes=height * width).squeeze(0)
        coords_current = torch.tensor(
            [(i // width, i % width) for i in range(height * width)],
            dtype=torch.float32,
            device=device
        )
        A_motif_curr = A_motif

        for block in self.dsr_blocks:
            x, A_current, block_diags, reg_loss = block(x, A_current, A_motif_curr, coords_current)
            A_motif_curr = block_diags['S'].t() @ A_motif_curr @ block_diags['S']
            block_diags['input_coords'] = coords_current.detach()
            diagnostics.append(block_diags)
            total_reg_loss += reg_loss
            coords_current = block_diags['coords']

        graph_embedding = x.mean(dim=0)
        predicted_score = self.final_mlp(graph_embedding)

        return predicted_score, total_reg_loss, diagnostics

    @torch.no_grad()
    def _node_features(self, board_np):
        height, width = board_np.shape
        black = (board_np == 1).astype(np.float32).reshape(-1, 1)
        white = (board_np == 2).astype(np.float32).reshape(-1, 1)
        empty = (board_np == 0).astype(np.float32).reshape(-1, 1)
        rows = np.arange(height).repeat(width).astype(np.float32) / max(1., height - 1)
        cols = np.tile(np.arange(width), height).astype(np.float32) / max(1., width - 1)
        rc = np.stack([rows, cols], axis=1)
        return torch.from_numpy(np.concatenate([black, white, empty, rc], axis=1))


class GoGame:
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.move_history = []
        self.current_player = 1
        self.graph = nx.grid_2d_graph(size, size)
        self.fig = None
        self.ax = None
        self.pyg_diagnostics = None
        self.active_layer_idx = 0
        self.model = None
        self.show_clusters = False
        self.artist_collections = defaultdict(list)

    def get_group(self, board, x, y):
        color = board[x, y]
        if color == 0:
            return set(), set()

        queue = [(x, y)]
        group = set()
        liberties = set()
        visited = {(x, y)}

        while queue:
            current_x, current_y = queue.pop(0)
            group.add((current_x, current_y))

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x, next_y = current_x + dx, current_y + dy
                if (0 <= next_x < self.size and
                        0 <= next_y < self.size and
                        (next_x, next_y) not in visited):
                    visited.add((next_x, next_y))
                    if board[next_x, next_y] == 0:
                        liberties.add((next_x, next_y))
                    elif board[next_x, next_y] == color:
                        queue.append((next_x, next_y))

        return group, liberties

    def is_valid_move(self, x, y):
        if self.board[x, y] != 0:
            return False

        board_copy = self.board.copy()
        board_copy[x, y] = self.current_player
        _, libs = self.get_group(board_copy, x, y)

        if libs:
            return True

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            if (0 <= next_x < self.size and
                    0 <= next_y < self.size and
                    board_copy[next_x, next_y] == 3 - self.current_player):
                _, opp_libs = self.get_group(board_copy, next_x, next_y)
                if not opp_libs:
                    return True
        return False

    def play_move(self, x, y):
        if not self.is_valid_move(x, y):
            return False

        self.move_history.append((self.current_player, self.board.copy()))
        self.board[x, y] = self.current_player
        opponent_color = 3 - self.current_player

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == opponent_color:
                    group, libs = self.get_group(self.board, i, j)
                    if not libs:
                        for stone_x, stone_y in group:
                            self.board[stone_x, stone_y] = 0

        self.current_player = 3 - self.current_player
        self.pyg_diagnostics = None
        return True

    def undo_move(self):
        if not self.move_history:
            return False

        last_player, last_board = self.move_history.pop()
        self.board = last_board.copy()
        self.current_player = last_player
        self.pyg_diagnostics = None
        self._update_visualization()
        return True

    def train_model(self, num_epochs=50, lr=0.005, anneal_factor=0.95):
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GoPyG_DSR_Model(size=self.size).to(device)

        for block in self.model.dsr_blocks:
            block.tau_anneal_factor = anneal_factor

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        target = float(np.sum(self.board == 1) - np.sum(self.board == 2))

        print(f"Training for {num_epochs} epochs... Target: {target:.1f}")

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()

            pred, reg_loss, _ = self.model(self.board)
            task_loss = F.mse_loss(pred.squeeze(), torch.tensor(target, device=device))
            total_loss = task_loss + reg_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                for block in self.model.dsr_blocks:
                    block.tau.mul_(block.tau_anneal_factor).clamp_(min=0.1)

            if (epoch + 1) % 10 == 0:
                print(
                    f"E {epoch + 1}: Loss={total_loss:.3f}(Task:{task_loss:.3f},Reg:{reg_loss:.3f}) Tau={self.model.dsr_blocks[0].tau.item():.3f}")

        print("Training complete!")

    def run_inference(self):
        if not self.model:
            print("Model not trained. Press 't'.")
            return

        self.model.eval()
        with torch.no_grad():
            _, _, diagnostics = self.model(self.board)

        self.pyg_diagnostics = diagnostics
        self.active_layer_idx = 0
        self.show_clusters = True

        print(f"Inference done. {len(diagnostics)} layers. Keys: [1]/[2].. view, [c] toggle clusters.")
        self._update_visualization()

    def _setup_visualization(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.pos = {(i, j): (j, -i) for i in range(self.size) for j in range(self.size)}

    def _update_visualization(self):
        if self.fig is None:
            self._setup_visualization()

        self.ax.cla()
        nx.draw_networkx_edges(self.graph, self.pos, edge_color='grey', alpha=0.2, ax=self.ax)

        for row in range(self.size):
            for col in range(self.size):
                color = 'black' if self.board[row, col] == 1 else 'white' if self.board[row, col] == 2 else '#D2B48C'
                self.ax.add_patch(Circle(self.pos[(row, col)], 0.45, color=color, zorder=2, ec='black'))

        title = f"Go {self.size}x{self.size} | Player: {'B' if self.current_player == 1 else 'W'} to move"
        legend_handles = [
            mpatches.Patch(color='black', label='Black'),
            mpatches.Patch(facecolor='white', edgecolor='black', label='White')
        ]

        if self.pyg_diagnostics:
            title += f" | Viewing DSR Layer {self.active_layer_idx + 1}"
            diagnostics = self.pyg_diagnostics[self.active_layer_idx]
            pos_map = diagnostics['input_coords'].cpu().numpy()
            xy = np.stack([pos_map[:, 1], -pos_map[:, 0]], axis=1)

            edges = set()
            for u, v in diagnostics['kept_edge_index'].cpu().numpy().T:
                if u == v:
                    continue
                edges.add(((u, v) if u < v else (v, u), 'green', '--'))

            for u, v in diagnostics['added_edge_index'].cpu().numpy().T:
                if u == v:
                    continue
                edges.add(((u, v) if u < v else (v, u), 'blue', ':'))

            for (u, v), color, linestyle in edges:
                x_coords = [xy[u, 0], xy[v, 0]]
                y_coords = [xy[u, 1], xy[v, 1]]
                self.ax.plot(x_coords, y_coords, color=color, lw=1.5, alpha=0.8, linestyle=linestyle)

            if self.show_clusters:
                S = diagnostics['S'].cpu().numpy()
                K = S.shape[1]
                colors = plt.cm.tab20(np.linspace(0, 1, K))

                for n in range(S.shape[0]):
                    cluster_idx = int(S[n].argmax())
                    x, y = xy[n]
                    self.ax.add_patch(Circle(
                        (x, y),
                        0.28,
                        lw=2.,
                        fill=False,
                        alpha=.8,
                        edgecolor=colors[cluster_idx]
                    ))

            legend_handles.extend([
                Line2D([0], [0], color='green', linestyle='--', label='Kept Edges'),
                Line2D([0], [0], color='blue', linestyle=':', label='Added Edges')
            ])

        self.ax.set_title(title)
        self.ax.set_aspect('equal')
        self.ax.legend(handles=legend_handles)
        plt.tight_layout()
        self.fig.canvas.draw_idle()

    def start(self):
        self._setup_visualization()
        self._update_visualization()
        plt.show(block=True)


def demo():
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    game = GoGame(size=9)
    moves = [(4, 4), (3, 3), (4, 2), (5, 5), (2, 4), (3, 5), (6, 4), (5, 3)]

    for row, col in moves:
        game.play_move(row, col)

    game.train_model(num_epochs=500)
    game.run_inference()
    game._setup_visualization()

    for i in range(len(game.pyg_diagnostics)):
        game.active_layer_idx = i
        game.show_clusters = True
        game._update_visualization()
        plt.savefig(f"go_dsr_layer_{i + 1}_clusters.png", dpi=300, bbox_inches='tight')
        game.show_clusters = False
        game._update_visualization()
        plt.savefig(f"go_dsr_layer_{i + 1}_graph.png", dpi=300, bbox_inches='tight')
        print(f"Saved layer {i + 1} visualizations.")

    plt.close(game.fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        GoGame(size=9).start()
    else:
        demo()
