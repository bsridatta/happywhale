import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.cuda import device_count
from dataset import Whales
from trainer import WhaleNet, BackboneFreeze
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def main():
    parser = get_argparser()
    opt = parser.parse_args()
    pl.seed_everything(opt.seed)

    train_loader = torch.utils.data.DataLoader(
        Whales(
            folds=[0, 1, 2, 3],
            is_train=True,
            **vars(opt),
        ),
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        Whales(folds=[4], is_train=True, **vars(opt)),
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        shuffle=False,
    )

    logger = WandbLogger(project="whale_kaggle", offline=False, log_model="all")
    logger.log_hyperparams(opt)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./ckpts/",
        filename=f"{logger.experiment.name}_" + "{epoch}-{val_loss:.2f}",
        mode="min",
        verbose=True,
    )
    checkpoint_callback.FILE_EXTENSION = ".pt"
    backbone_freeze = BackboneFreeze(train_bn=False)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        gpus=device_count() * int(opt.use_gpu),
        fast_dev_run=opt.fast_dev_run,
        max_epochs=opt.epochs,
        callbacks=[checkpoint_callback, backbone_freeze, lr_monitor],
        logger=logger,
        precision=opt.precision,
        auto_lr_find=True,
        # resume_from_checkpoint=opt.resume_ckpt,
        # log_every_n_steps=1,
    )

    model = WhaleNet(opt)
    if opt.load_weights:
        ckpt = torch.load(opt.load_weights)
        model.load_state_dict(ckpt['state_dict'])

    if not opt.eval_run:
        trainer.fit(model, train_loader, val_loader)
    else:
        # trainer.evaluating
        pass

    print(
        f"Best model @ {trainer.checkpoint_callback.best_model_score}: {trainer.checkpoint_callback.best_model_path}"
    )


def get_argparser():
    parser = ArgumentParser()

    # fmt: off
    # training specific
    parser.add_argument('--epochs', default=2, type=int,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='number of samples per step, have more than one for batch norm')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='learning rate')
    parser.add_argument('--fast_dev_run', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='run all methods once to check integrity')
    parser.add_argument('--img_size', default=512, type=int,
                        help='change image to img_size before passing it as input to the model')
    parser.add_argument('--n_class', default=15587, type=int,
                        help='class in the training data')
    parser.add_argument('--embedding_size', default=512, type=int,
                        help='final embedding')
    parser.add_argument('--precision', default=32, type=int,
                        help='flot precision')


    # data files
    parser.add_argument('--data_root', default=f"{os.environ['HOME']}/lab/data/", type=str,
                        help='abs path to training data')
    parser.add_argument('--train_image_path', default=f"train_images/", type=str,
                        help='path')
    parser.add_argument('--test_image_path', default=f"test_images", type=str,
                        help='path')
    parser.add_argument('--train_csv_path', default=f"train_equal_species_ids.csv", type=str,
                        help='path')
    parser.add_argument('--test_csv_path', default=f"sample_submission.csv", type=str,
                        help='path')

    parser.add_argument('--resume_ckpt', default=None, type=str,
                        help='wandb checkpoint reference')

    parser.add_argument('--load_weights', default=None, type=str,
                        help='wandb checkpoint reference')

    parser.add_argument('--eval_run', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='enable gpu if available')
    # output
    # device
    parser.add_argument('--use_gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='enable gpu if available')
    parser.add_argument('--pin_memory', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='pin memory to device')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='workers for data loader')
    parser.add_argument('--seed', default=400, type=int,
                        help='random seed')

    # fmt: on
    return parser


if __name__ == "__main__":
    main()
