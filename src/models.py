from typing import Type
import torch.nn as nn
import torch
from utils import GeM
import torchvision.models as models
import timm


class EfficientNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b4",
        pretrained: bool = True,
        activation: Type[nn.Module] = nn.ReLU,
        p_dropout: float = 0.5,
        embedding_size: int = 200,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
        )

        out_features_backbone = self.backbone.classifier.in_features
        # remove global_pool and classifier layers
        self.backbone.global_pool = nn.Identity()
        self.backbone.classifier = nn.Identity()

        self.pooling = GeM()
        # add extra FC layer
        self.embeddings = nn.Sequential(
            # nn.Linear(out_features_backbone, out_features_backbone),
            # nn.BatchNorm1d(out_features_backbone),
            # activation(),
            # nn.Dropout(p=p_dropout),
            nn.Linear(out_features_backbone, embedding_size),
            # nn.BatchNorm1d(embedding_size),
            # activation(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x).flatten(1)
        x = self.embeddings(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    net = EfficientNet()
    # summary(net, input_size=(4, 3, 500, 500))
    # for param in net.parameters():
    #     if not param.requires_grad:
    #         print("*")
    print(net)