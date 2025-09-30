import copy
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from go_motif_pooling.models import MPoolModel
from go_motif_pooling.data import GoBoardGraphDataset

class Trainer:
    def __init__(
        self,
        model: MPoolModel,
        dataset: GoBoardGraphDataset,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.dataset = dataset
        self.config = config or {}

        self.data_splits = {}
        self.loaders = {}
        self.training_history = []
        self.optimizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def setup_data_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 8,
        seed: int = 0
    ):
        total_samples = len(self.dataset)

        train_len = max(1, int(math.floor(total_samples * train_ratio)))
        val_len = max(1, int(math.floor(total_samples * val_ratio)))
        test_len = total_samples - train_len - val_len

        if test_len <= 0:
            test_len = 1
            if train_len > val_len:
                train_len = max(1, train_len - 1)
            else:
                val_len = max(1, val_len - 1)

        generator = torch.Generator().manual_seed(seed)
        train_set, val_set, test_set = random_split(
            self.dataset, [train_len, val_len, test_len], generator=generator
        )

        train_loader = DataLoader(
            train_set, batch_size=min(batch_size, train_len), shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=min(batch_size, val_len), shuffle=False
        )
        test_loader = DataLoader(
            test_set, batch_size=min(batch_size, test_len), shuffle=False
        )

        self.data_splits = {'train': train_set, 'val': val_set, 'test': test_set}
        self.loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        print(f'Data split: train={train_len}, val={val_len}, test={test_len}')

    def evaluate_loader(self, loader: DataLoader) -> Optional[Dict[str, float]]:
        self.model.eval()
        total_task = 0.0
        total_reg = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred, reg_loss, _ = self.model(batch)
                pred = pred.view(-1)
                targets = batch.y.to(self.device).view(-1)

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

    def train(
        self,
        num_epochs: int = 100,
        batch_size: int = 8,
        lr: float = 0.005,
        patience: int = 20,
        seed: int = 0
    ) -> float:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if not self.loaders:
            self.setup_data_splits(batch_size=batch_size, seed=seed)

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float('inf')
        patience_counter = 0
        self.training_history.clear()
        train_loader = self.loaders['train']
        val_loader = self.loaders['val']

        print(f'Training for up to {num_epochs} epochs on {len(self.dataset)} samples')
        print(f'Device: {self.device}')

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            batches = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                pred, reg_loss, _ = self.model(batch)
                pred = pred.view(-1)
                targets = batch.y.to(self.device).view(-1)

                task_loss = F.mse_loss(pred, targets)
                reg_term = reg_loss.mean()
                loss = task_loss + reg_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            avg_train_loss = epoch_loss / max(1, batches)
            val_metrics = self.evaluate_loader(val_loader)
            val_total = val_metrics['total'] if val_metrics else float('inf')

            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_total': val_total
            })

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}: Train={avg_train_loss:.3f}, Val={val_total:.3f}')

            if val_total < best_val:
                best_val = val_total
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        self.model.load_state_dict(best_state)
        print('Training complete!')
        return best_val

    def evaluate(self, split: str = 'test') -> Dict[str, float]:
        if split not in self.loaders:
            raise ValueError(f'Unknown split: {split}. Available: {list(self.loaders.keys())}')

        loader = self.loaders[split]
        metrics = self.evaluate_loader(loader)
        return metrics if metrics else {}

    def run_inference(
        self,
        split: str = 'test',
        limit: Optional[int] = None
    ) -> List[Dict]:
        if split not in self.loaders:
            raise ValueError(f'Unknown split: {split}')

        loader = self.loaders[split]
        results = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
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

        return results

    def get_training_history(self) -> List[Dict]:
        return self.training_history

    def save_checkpoint(self, path: str):
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'training_history': self.training_history,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f'Checkpoint saved to {path}')

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])

        if self.optimizer and checkpoint.get('optimizer_state'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        print(f'Checkpoint loaded from {path}')
