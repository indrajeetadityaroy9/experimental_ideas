from openhgnn import Experiment
import torch

gpu = 0 if torch.cuda.is_available() else -1

exp = Experiment(
    model='RGCN',
    dataset='acm4GTN',
    task='node_classification',
    gpu=gpu,
    lr=0.005,
    hidden_dim=128,
    num_layers=2,
    dropout=0.5,
    n_bases=30,
    weight_decay=5e-4,
    fanout=10,               
    mini_batch_flag=True,
    use_self_loop=True,
    max_epoch=50,
    patience=10,
    use_distributed=False,
    graphbolt=False,
    use_best_config=False,
    seed=0,
)
print(exp.run())