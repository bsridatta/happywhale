import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path

class GeM(nn.Module):
    """Source: https://amaarora.github.io/2020/08/30/gempool.html"""

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"GeM p={float(self.p.data):.4f}, eps={str(self.eps)}"


    # if opt.wandb_ckpt:
    #     logger.experiment.log_artifact
    #     artifact = logger.experiment.use_artifact("b-sridatta/whale_kaggle/" + run, type="model")
    #     ckpt_path = Path.joinpath(artifact.download(), "model.ckpt")
    # logger.watch(model, log='all')
