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


# src: https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py


class ArcMarginProduct(nn.Module):
    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0
    ):
        """
        in_features: dimension of the input
        out_features: dimension of the last layer (in our case the classification)
        s: norm of input feature
        m: margin
        ls_eps: label smoothing"""

        super(ArcMarginProduct, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # Fills the input `Tensor` with values according to the method described in
        # `Understanding the difficulty of training deep feedforward neural networks`
        # Glorot, X. & Bengio, Y. (2010)
        # using a uniform distribution.
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        one_hot = torch.zeros(cosine.size()).to(device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    # if opt.wandb_ckpt:
    #     logger.experiment.log_artifact
    #     artifact = logger.experiment.use_artifact("b-sridatta/whale_kaggle/" + run, type="model")
    #     ckpt_path = Path.joinpath(artifact.download(), "model.ckpt")
    # logger.watch(model, log='all')
