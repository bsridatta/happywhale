from asyncio.log import logger
import pytorch_lightning as pl
import torch
from models import EfficientNet
from pytorch_metric_learning import losses
from pytorch_lightning.callbacks.finetuning import BaseFinetuning


class WhaleNet(pl.LightningModule):
    def __init__(self, opt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.opt = opt
        self.model = EfficientNet(embedding_size=opt.embedding_size)
        self.loss_fn = losses.ArcFaceLoss(
            embedding_size=opt.embedding_size, num_classes=opt.n_class
        )  # TODO: class weights

    def forward(self, x):
        x = self.model(x)  # mostly for inference
        return x

    def training_step(self, batch, batch_idx):
        inp, label = batch["image"], batch["individual_id"]
        out = self.model(inp)  # embedding
        loss = self.loss_fn(out, label)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        inp, label = batch["image"], batch["individual_id"]
        out = self.model(inp)  # embedding
        loss = self.loss_fn(out, label)
        self.log("val_loss", loss, on_epoch=True, batch_size=len(batch))
        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.opt.lr
        )

    # def test_step(self, *args, **kwargs):
    #     return None

class BackboneFreeze(BaseFinetuning):
    def __init__(self, train_bn=False):
        super().__init__()
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(modules=pl_module.model.backbone, train_bn=self.train_bn)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        pass
