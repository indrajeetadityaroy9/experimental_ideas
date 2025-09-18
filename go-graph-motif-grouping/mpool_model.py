import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
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


def board_one_hot(board_np, device=None):
    height, width = board_np.shape
    kwargs = {'device': device} if device is not None else {}
    black = torch.tensor((board_np == 1).astype(np.float32), **kwargs).view(1, 1, height, width)
    white = torch.tensor((board_np == 2).astype(np.float32), **kwargs).view(1, 1, height, width)
    empty = torch.tensor((board_np == 0).astype(np.float32), **kwargs).view(1, 1, height, width)
    return black, white, empty


def _conv_hits(x, kernel):
    return F.conv2d(x, kernel, stride=1, padding=0)


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


class GoBoardGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


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

        device = self.k_diag.device
        black, white, empty = board_one_hot(board_np, device=device)

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


def readout_features(x):
    if x.numel() == 0:
        raise ValueError('Cannot readout from empty feature matrix')
    mean = x.mean(dim=0)
    max_vals = x.max(dim=0).values
    return torch.cat([mean, max_vals], dim=0)


class MPoolModel(nn.Module):
    def __init__(self, size=9, in_dim=5, hidden=64, num_layers=2, pool_ratios=None, cluster_ratios=None,
                 lambda_cut=1.0, lambda_ortho=1.0):
        super().__init__()
        if pool_ratios is None:
            pool_ratios = [0.5] * num_layers
        if cluster_ratios is None:
            cluster_ratios = [0.5] * num_layers
        assert len(pool_ratios) == num_layers
        assert len(cluster_ratios) == num_layers

        self.size = size
        self.hidden = hidden
        self.num_layers = num_layers
        self.pool_ratios = pool_ratios
        self.cluster_ratios = cluster_ratios
        self.lambda_cut = lambda_cut
        self.lambda_ortho = lambda_ortho

        self.mconv = MotifConv2D()
        self.initial_feature_map = nn.Linear(in_dim, hidden)

        self.selection_pools = nn.ModuleList()
        self.clustering_pools = nn.ModuleList()

        current_nodes = size * size
        for i in range(num_layers):
            pool_ratio = pool_ratios[i]
            cluster_ratio = cluster_ratios[i]
            clusters = max(2, int(current_nodes * cluster_ratio))
            self.selection_pools.append(MotifSelectionPool(hidden, pool_ratio))
            self.clustering_pools.append(MotifClusteringPool(hidden, hidden, clusters))
            current_nodes = clusters

        readout_dim = hidden * 2 * num_layers * 2
        self.final_mlp = nn.Sequential(
            nn.Linear(readout_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, data):
        device = self.initial_feature_map.weight.device
        if isinstance(data, Batch):
            graphs = data.to_data_list()
        elif isinstance(data, Data):
            graphs = [data]
        elif isinstance(data, (list, tuple)):
            graphs = list(data)
        else:
            raise TypeError('MPoolModel.forward expected a PyG Data, Batch, or list of Data objects.')

        predictions = []
        reg_losses = []
        diagnostics = []

        for graph in graphs:
            pred, reg_loss, diag = self._forward_single_graph(graph, device)
            predictions.append(pred)
            reg_losses.append(reg_loss)
            diagnostics.append(diag)

        predictions = torch.stack(predictions, dim=0)
        reg_losses = torch.stack(reg_losses, dim=0)
        return predictions, reg_losses, diagnostics

    def _forward_single_graph(self, graph, device):
        graph = graph.to(device)
        x = graph.x.float()
        num_nodes = x.size(0)
        if num_nodes == 0:
            raise ValueError('Cannot run pooling on an empty graph')

        edge_weight = getattr(graph, 'edge_weight', None)
        adjacency = to_dense_adj(graph.edge_index, max_num_nodes=num_nodes, edge_attr=edge_weight).squeeze(0)
        adjacency = adjacency.to(device=device, dtype=torch.float32)

        if hasattr(graph, 'motif_edge_index'):
            motif_weight = getattr(graph, 'motif_edge_weight', None)
            motif_adjacency = to_dense_adj(
                graph.motif_edge_index, max_num_nodes=num_nodes, edge_attr=motif_weight
            ).squeeze(0).to(device=device, dtype=torch.float32)
        else:
            if not hasattr(graph, 'board_state') or not hasattr(graph, 'board_size'):
                raise ValueError('Motif information missing. Provide motif_edge_index or board metadata.')
            height, width = graph.board_size.tolist()
            board_np = graph.board_state.view(height, width).detach().cpu().numpy()
            motif_hits = self.mconv(board_np)
            motif_adjacency = motif_adjacency_from_hits(motif_hits, height, width).to(device)

        x0 = self.initial_feature_map(x)

        sel_x = x0
        sel_adj = adjacency
        sel_motif = motif_adjacency

        clu_x = x0
        clu_adj = adjacency
        clu_motif = motif_adjacency

        selection_readouts = []
        clustering_readouts = []
        diagnostics = []
        total_reg_loss = torch.tensor(0.0, device=device)

        for layer_idx in range(self.num_layers):
            sel_pool = self.selection_pools[layer_idx]
            sel_out = sel_pool(sel_x, sel_adj, sel_motif)
            sel_x = sel_out['x']
            sel_adj = sel_out['adjacency']
            sel_motif = sel_out['motif_adjacency']
            selection_readouts.append(readout_features(sel_x))

            clu_pool = self.clustering_pools[layer_idx]
            clu_out = clu_pool(clu_x, clu_adj, clu_motif)
            clu_x = clu_out['x']
            clu_adj = clu_out['adjacency']
            clu_motif = clu_out['motif_adjacency']
            clustering_readouts.append(readout_features(clu_x))

            losses = clu_out['losses']
            reg_loss = self.lambda_cut * losses['cut'] + self.lambda_ortho * losses['ortho']
            total_reg_loss = total_reg_loss + reg_loss

            diagnostics.append({
                'layer': layer_idx + 1,
                'selection': {
                    'scores': sel_out['scores'].detach().cpu(),
                    'kept_indices': sel_out['indices'].detach().cpu()
                },
                'clustering': {
                    'S': clu_out['S'].cpu(),
                    'losses': {k: v.detach().cpu().item() for k, v in losses.items()}
                }
            })

        selection_stack = torch.cat(selection_readouts, dim=0)
        clustering_stack = torch.cat(clustering_readouts, dim=0)
        combined = torch.cat([selection_stack, clustering_stack], dim=0)
        prediction = self.final_mlp(combined).squeeze()

        return prediction, total_reg_loss, diagnostics


class MPoolExperiment:
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.move_history = []
        self.current_player = 1
        self.pyg_diagnostics = None
        self.model = None
        self.data_splits = {}
        self.loaders = {}
        self.training_history = []

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
                if (0 <= next_x < self.size and 0 <= next_y < self.size
                        and (next_x, next_y) not in visited):
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
            if (0 <= next_x < self.size and 0 <= next_y < self.size
                    and board_copy[next_x, next_y] == 3 - self.current_player):
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
        return True

    def _board_target(self, board):
        return float(np.sum(board == 1) - np.sum(board == 2))

    def _gather_boards(self):
        states = [board.copy() for _, board in self.move_history]
        states.append(self.board.copy())
        return states

    def _build_dataset(self):
        boards = self._gather_boards()
        motif_conv = MotifConv2D()
        graphs = [board_to_data(board, target=self._board_target(board), motif_conv=motif_conv)
                  for board in boards]
        return GoBoardGraphDataset(graphs)

    def _split_lengths(self, total_samples):
        train_len = max(1, int(math.floor(total_samples * 0.8)))
        val_len = max(1, int(math.floor(total_samples * 0.1)))
        remaining = total_samples - train_len - val_len
        if remaining <= 0:
            remaining = 1
            if train_len > val_len:
                train_len = max(1, train_len - 1)
            else:
                val_len = max(1, val_len - 1)
        return train_len, val_len, remaining

    def _evaluate_loader(self, loader, device):
        self.model.eval()
        total_task = 0.0
        total_reg = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred, reg_loss, _ = self.model(batch)
                pred = pred.view(-1)
                targets = batch.y.to(device).view(-1)
                task_loss = F.mse_loss(pred, targets, reduction='sum')
                total_task += task_loss.item()
                total_reg += reg_loss.sum().item()
                total_examples += targets.numel()

        if total_examples == 0:
            return None

        return {
            'task': total_task / total_examples,
            'reg': total_reg / total_examples,
            'total': (total_task + total_reg) / total_examples
        }

    def train_model(self, num_epochs=100, batch_size=8, lr=0.005, patience=20, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        dataset = self._build_dataset()
        total_samples = len(dataset)
        train_len, val_len, test_len = self._split_lengths(total_samples)

        generator = torch.Generator().manual_seed(seed)
        train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

        train_loader = DataLoader(train_set, batch_size=min(batch_size, train_len), shuffle=True)
        val_loader = DataLoader(val_set, batch_size=min(batch_size, val_len), shuffle=False)
        test_loader = DataLoader(test_set, batch_size=min(batch_size, test_len), shuffle=False)

        self.data_splits = {'train': train_set, 'val': val_set, 'test': test_set}
        self.loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MPoolModel(size=self.size).to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float('inf')
        patience_counter = 0
        self.training_history.clear()

        print(f'Training for up to {num_epochs} epochs on {total_samples} samples '
              f'(train/val/test = {train_len}/{val_len}/{test_len})')

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            batches = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred, reg_loss, _ = self.model(batch)
                pred = pred.view(-1)
                targets = batch.y.to(device).view(-1)
                task_loss = F.mse_loss(pred, targets)
                reg_term = reg_loss.mean()
                loss = task_loss + reg_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            avg_train_loss = epoch_loss / max(1, batches)
            val_metrics = self._evaluate_loader(val_loader, device)
            val_total = val_metrics['total'] if val_metrics else float('inf')

            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_total': val_total
            })

            if (epoch + 1) % 10 == 0:
                print(f'E {epoch + 1}: Train={avg_train_loss:.3f}, Val={val_total:.3f}')

            if val_total < best_val:
                best_val = val_total
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break

        self.model.load_state_dict(best_state)
        print('Training complete!')
        return best_val

    def run_inference(self, split='test', limit=None):
        if not self.model:
            raise RuntimeError('Model not trained. Call train_model first.')
        if split not in self.loaders:
            raise ValueError(f'Unknown split: {split}')

        device = next(self.model.parameters()).device
        loader = self.loaders[split]
        results = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred, reg_loss, diagnostics = self.model(batch)
                for idx in range(pred.size(0)):
                    results.append({
                        'prediction': pred[idx].item(),
                        'reg_loss': reg_loss[idx].item(),
                        'diagnostics': diagnostics[idx]
                    })
                    if limit is not None and len(results) >= limit:
                        break
                if limit is not None and len(results) >= limit:
                    break

        self.pyg_diagnostics = results
        return results

def demo(num_epochs=200, board_setup=None):
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    experiment = MPoolExperiment(size=9)

    if board_setup is None:
        board_setup = [(4, 4), (3, 3), (4, 2), (5, 5), (2, 4), (3, 5), (6, 4), (5, 3)]

    for row, col in board_setup:
        experiment.play_move(row, col)

    experiment.train_model(num_epochs=num_epochs)
    diagnostics = experiment.run_inference(split='test', limit=1)

    if diagnostics:
        for layer_idx, layer_diag in enumerate(diagnostics[0]['diagnostics'], start=1):
            losses = layer_diag['clustering']['losses']
            print(f"Layer {layer_idx}: cut={losses['cut']:.4f}, ortho={losses['ortho']:.4f}")
    return diagnostics


if __name__ == "__main__":
    demo()
