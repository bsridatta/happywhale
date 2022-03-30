import pytorch_lightning as pl
import torchmetrics
from models import EfficientNet
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from utils import ArcMarginProduct
from torch.optim import Adam, lr_scheduler
from timm.optim import create_optimizer_v2
import torch.nn.functional as F


class WhaleNet(pl.LightningModule):
    def __init__(self, opt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.opt = opt
        self.model = EfficientNet(embedding_size=opt.embedding_size)
        self.loss_fn = F.cross_entropy
        self.train_acc = torchmetrics.Accuracy(
            average="macro", num_classes=self.opt.n_class
        )
        self.val_acc = torchmetrics.Accuracy(
            average="macro", num_classes=self.opt.n_class
        )

        # TODO: class weights
        self.arcface = ArcMarginProduct(
            in_features=opt.embedding_size,
            out_features=opt.n_class,
            s=opt.scale,
            m=opt.margin,
            easy_margin=opt.easy_margin,
            ls_eps=opt.ls_eps,
        )

    def forward(self, x):
        x = self.model(x)  # mostly for inference
        return x

    def training_step(self, batch, batch_idx):
        inp, label = batch["image"], batch["individual_id"]
        emb = self.model(inp)  # embedding
        out = self.arcface(emb, label, self.device)
        loss = self.loss_fn(out, label)

        log_dict = {
            "train_loss": loss,
            "train_acc": self.train_acc(out, label),
        }
        self.log_dict(log_dict, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        inp, label = batch["image"], batch["individual_id"]
        emb = self.model(inp)  # embedding
        out = self.arcface(emb, label, self.device)
        loss = self.loss_fn(out, label)

        log_dict = {
            "val_loss": loss,
            "val_acc": self.val_acc(out, label),
        }
        self.log_dict(log_dict, batch_size=len(batch))
        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        # optimizer = Adam(
        #     filter(lambda p: p.requires_grad, self.parameters()),
        #     lr=self.opt.lr,
        #     # weight_decay=self.opt.weight_decay,
        # )
        # optimizer = create_optimizer_v2(
        #     self.parameters(),
        #     opt="adam",
        #     lr=self.opt.lr,
        #     weight_decay=self.opt.weight_decay,
        # )
        optimizer = Adam(
            self.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay
        )
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            self.opt.lr,
            steps_per_epoch=self.opt.len_train_loader,
            epochs=self.opt.epochs,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]
        # scheduler = {
        #     "scheduler": lr_scheduler.OneCycleLR(
        #         optimizer,
        #         self.opt.lr,
        #         steps_per_epoch=self.opt.len_train_loader,
        #         epochs=self.opt.epochs,
        #     ),
        #     "interval": "step",
        # }

        # return optimizer  # [optimizer]  # , [scheduler]

    def predict_step(self, batch, batch_idx):

        inp = batch["image"]
        embeddings = self.model(inp)

        output = {"embeddings": embeddings}

        if "individual_id_org" in batch:
            output["labels"] = batch["individual_id"]

        return output


class BackboneFreeze(BaseFinetuning):
    def __init__(self, train_bn=False):
        super().__init__()
        self.train_bn = train_bn

    # TODO
    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(modules=pl_module.model.backbone, train_bn=self.train_bn)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        pass
