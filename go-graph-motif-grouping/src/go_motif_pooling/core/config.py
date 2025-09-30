import os
import yaml
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

@dataclass
class ModelConfig:
    name: str = "MPoolModel"
    board_size: int = 9
    in_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    pool_ratios: list = field(default_factory=lambda: [0.5, 0.5])
    cluster_ratios: list = field(default_factory=lambda: [0.5, 0.5])
    lambda_cut: float = 1.0
    lambda_ortho: float = 1.0

@dataclass
class DataConfig:
    dataset_type: str = "hybrid"
    num_tactical: int = 60
    num_game: int = 40
    complexity: str = "medium"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True

@dataclass
class TrainingConfig:
    max_epochs: int = 150
    patience: int = 25
    grad_clip: float = 1.0
    log_interval: int = 10
    save_interval: int = 10
    val_interval: int = 1
    resume: Optional[str] = None

@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    eps: float = 1e-8

@dataclass
class SchedulerConfig:
    name: str = "reduce_on_plateau"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 10
    min_lr: float = 1e-6
    verbose: bool = True

@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "go-motif-pooling"
    wandb_entity: Optional[str] = None
    use_tensorboard: bool = True
    log_metrics: list = field(default_factory=lambda: ["loss", "task_loss", "reg_loss", "mse", "mae", "r2"])
    log_gradients: bool = False
    log_parameters: bool = False

@dataclass
class EvaluationConfig:
    metrics: list = field(default_factory=lambda: ["mse", "mae", "rmse", "r2"])
    save_predictions: bool = True
    visualize: bool = True

@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        return cls(
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {})),
            scheduler=SchedulerConfig(**config_dict.get('scheduler', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def update(self, updates: Dict[str, Any]) -> 'Config':
        config_dict = self.to_dict()
        _deep_update(config_dict, updates)
        return Config.from_dict(config_dict)

def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    default_config_path = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"

    if os.path.exists(default_config_path):
        config = Config.from_yaml(str(default_config_path))
    else:
        config = Config()

    if config_path and os.path.exists(config_path):
        specified_config = Config.from_yaml(config_path)
        config = config.update(specified_config.to_dict())

    if overrides:
        config = config.update(overrides)

    return config

def get_default_config() -> Config:
    return Config()
