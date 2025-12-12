import lightning.pytorch as pl


class BaseLightningModule(pl.LightningModule):
    """Base lightning module"""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        # self.preprocessors = {}
        self.val_names = {}

    def setup_datamodule(self):
        datamodule = self.trainer.datamodule
        # set validation set names
        self.val_names = [name for name in datamodule.val_dataset()]

    def on_fit_start(self):
        self.setup_datamodule()
