import numpy as np
import torch
from typing import Dict, List, Union

def compute_mse(predictions: Union[np.ndarray, torch.Tensor],targets: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    return float(np.mean((predictions - targets) ** 2))


def compute_mae(predictions: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    return float(np.mean(np.abs(predictions - targets)))


def compute_rmse(predictions: Union[np.ndarray, torch.Tensor],
                 targets: Union[np.ndarray, torch.Tensor]) -> float:
    mse = compute_mse(predictions, targets)
    return float(np.sqrt(mse))


def compute_r2(predictions: Union[np.ndarray, torch.Tensor],
               targets: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - (ss_res / ss_tot))


def compute_all_metrics(predictions: Union[np.ndarray, torch.Tensor],
                       targets: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    return {
        'mse': compute_mse(predictions, targets),
        'mae': compute_mae(predictions, targets),
        'rmse': compute_rmse(predictions, targets),
        'r2': compute_r2(predictions, targets),
    }


class MetricsTracker:
    def __init__(self):
        self.history = []

    def update(self, metrics: Dict[str, float], epoch: int):
        metrics_copy = metrics.copy()
        metrics_copy['epoch'] = epoch
        self.history.append(metrics_copy)

    def get_best(self, metric_name: str, mode: str = 'min') -> Dict[str, float]:
        if not self.history:
            return {}

        if mode == 'min':
            best = min(self.history, key=lambda x: x.get(metric_name, float('inf')))
        else:
            best = max(self.history, key=lambda x: x.get(metric_name, float('-inf')))

        return best

    def get_latest(self) -> Dict[str, float]:
        return self.history[-1] if self.history else {}

    def to_dict(self) -> List[Dict[str, float]]:
        return self.history
