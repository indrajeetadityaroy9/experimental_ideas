import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj
from go_motif_pooling.models.layers import MotifSelectionPool, MotifClusteringPool
from go_motif_pooling.game.patterns import MotifConv2D, motif_adjacency_from_hits
from go_motif_pooling.utils.graph import readout_features

class MPoolModel(nn.Module):
    def __init__(self, size=9, in_dim=5, hidden=64, num_layers=2,
                 pool_ratios=None, cluster_ratios=None,
                 lambda_cut=1.0, lambda_ortho=1.0):
        super().__init__()

        if pool_ratios is None:
            pool_ratios = [0.5] * num_layers
        if cluster_ratios is None:
            cluster_ratios = [0.5] * num_layers

        assert len(pool_ratios) == num_layers, "pool_ratios length must match num_layers"
        assert len(cluster_ratios) == num_layers, "cluster_ratios length must match num_layers"

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
            raise TypeError(
                'MPoolModel.forward expected a PyG Data, Batch, or list of Data objects. '
                f'Got {type(data)}'
            )

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
        adjacency = to_dense_adj(
            graph.edge_index,
            max_num_nodes=num_nodes,
            edge_attr=edge_weight
        ).squeeze(0)
        adjacency = adjacency.to(device=device, dtype=torch.float32)

        if hasattr(graph, 'motif_edge_index'):
            motif_weight = getattr(graph, 'motif_edge_weight', None)
            motif_adjacency = to_dense_adj(
                graph.motif_edge_index,
                max_num_nodes=num_nodes,
                edge_attr=motif_weight
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
