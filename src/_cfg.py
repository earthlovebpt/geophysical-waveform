
from types import SimpleNamespace
import torch

cfg= SimpleNamespace()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.local_rank = 0
cfg.seed = 123
cfg.subsample = None

cfg.backbone = "convnext_small.fb_in22k_ft_in1k"
cfg.ema = True
cfg.ema_decay = 0.99

cfg.epochs = 2
cfg.batch_size = 16
cfg.batch_size_val = 16

cfg.early_stopping = {"patience": 3, "streak": 0}
cfg.logging_steps = 100
