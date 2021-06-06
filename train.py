from pytorch_lightning.core import datamodule
import torch
from torch import nn
import pytorch_lightning as pl

from models import Unet
from datamodules import SuperResoNCDataModule


class Net(pl.LightningModule):
    def __init__(self, lr, batch_size, input_size, num_heights) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = Unet(5, 5, num_heights)
        self.criterion = nn.MSELoss()

    def forward(self, input_, height):
        return self.model(input_, height)

    def shared_step(self, batch):
        height, input_, target = batch
        pred = self(input_, height)
        loss = self.criterion(pred, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


if __name__ == '__main__':
    from rich.traceback import install
    install()
    pl.seed_everything(42)
    DATA_DIR = './data/processed/data.h5'
    BATCH_SIZE = 128
    USE_CV2 = False
    INPUT_SIZE = (5, 5)
    STEP = (3, 3)
    SHIFT = (1, 0)

    LR = 5e-4
    MAX_EPOCHS = 600

    datamodule = SuperResoNCDataModule(DATA_DIR, BATCH_SIZE, USE_CV2, INPUT_SIZE, STEP, SHIFT)
    model = Net(LR, BATCH_SIZE, INPUT_SIZE, 12)

    checkpoint = ''
    if not checkpoint:
        trainer = pl.Trainer(
            gpus=[9],
            deterministic=True,
            max_epochs=MAX_EPOCHS
        )
        trainer.fit(model, datamodule)

    