import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def board_one_hot(board_np, device=None):
    height, width = board_np.shape
    kwargs = {'device': device} if device is not None else {}
    black = torch.tensor((board_np == 1).astype(np.float32), **kwargs).view(1, 1, height, width)
    white = torch.tensor((board_np == 2).astype(np.float32), **kwargs).view(1, 1, height, width)
    empty = torch.tensor((board_np == 0).astype(np.float32), **kwargs).view(1, 1, height, width)
    return black, white, empty

def _conv_hits(x, kernel):
    return F.conv2d(x, kernel, stride=1, padding=0)

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

    weights = {
        'bamboo': 1.2,
        'tiger': 1.0,
        'empty_triangle': 0.8
    }

    for pattern, weight in weights.items():
        for color in ['black', 'white']:
            pattern_name = f'{pattern}_{color}'
            if pattern_name in hits:
                pattern_mask = hits[pattern_name]
                rows, cols = torch.where(pattern_mask)
                for row, col in zip(rows.tolist(), cols.tolist()):
                    add_2x2(row, col, weight)

    for color in ['black', 'white']:
        eye_name = f'eye_{color}'
        if eye_name in hits:
            eye_mask = hits[eye_name]
            rows, cols = torch.where(eye_mask)
            for row, col in zip(rows.tolist(), cols.tolist()):
                add_eye3x3(row, col, 1.5)

    return torch.maximum(adjacency, adjacency.t())
