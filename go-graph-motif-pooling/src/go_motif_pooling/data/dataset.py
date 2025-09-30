from torch.utils.data import Dataset

class GoBoardGraphDataset(Dataset):

    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __repr__(self):
        return f'GoBoardGraphDataset({len(self)} graphs)'
