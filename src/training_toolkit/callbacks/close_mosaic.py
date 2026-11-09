"""Callback to disable multi-sample augmentations in the final training epochs."""

import logging

import pytorch_lightning as L

logger = logging.getLogger(__name__)


class CloseMosaicCallback(L.Callback):
    """Disable mosaic, mixup, cutmix, and copy-paste for the last N epochs.

    Multi-sample augmentations (mosaic, mixup, cutmix, copy-paste) create
    training images with artificial compositions. Disabling them in the
    final epochs lets the model fine-tune on realistic single-image data,
    improving bbox regression and confidence calibration.

    Args:
        close_epoch: The epoch at which to disable augmentations.
            Computed as ``max_epochs - close_mosaic``.
    """

    def __init__(self, close_epoch: int) -> None:
        self.close_epoch = close_epoch
        self._closed = False

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
    ) -> None:
        if self._closed or trainer.current_epoch < self.close_epoch:
            return

        dataset = trainer.train_dataloader.dataset
        closed = self._close_augmentations(dataset)
        self._closed = True

        if closed:
            logger.info(
                "Epoch %d: disabled mosaic/mixup/cutmix/copy-paste for final epochs",
                trainer.current_epoch,
            )

    @staticmethod
    def _close_augmentations(dataset) -> bool:
        """Zero out multi-sample augmentation probabilities.

        The aug config may live on the dataset itself (tiled/streaming
        WRAPPERS store their own _aug_config) or on its _base (non-tiled
        datasets) — zero it wherever it is found. Zeroing both is idempotent
        when they alias the same AugConfig object.

        Returns:
            True if at least one config was zeroed; False if no config was found.
        """
        targets = [dataset]
        base = getattr(dataset, "_base", None)
        if base is not None:
            targets.append(base)
        closed = False
        for target in targets:
            aug = getattr(target, "_aug_config", None)
            if aug is not None:
                aug.mosaic_p = 0.0
                aug.mixup_p = 0.0
                aug.cutmix_p = 0.0
                aug.copy_paste_p = 0.0
                closed = True
        if not closed:
            logger.warning(
                "close_mosaic: no _aug_config found on the dataset or its _base "
                "— multi-sample augmentations were NOT disabled")
        return closed
