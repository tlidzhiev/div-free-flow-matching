import jax

from lab.logger.image import caption_image, make_grid

from .base import BaseTrainer
from .steps import Params


class Trainer(BaseTrainer):
    """
    Trainer for image classification models.
    """

    def _log_batch(self, params: Params, batch: dict[str, jax.Array], part: str) -> None:
        """
        Log a grid of the first images of the batch, each captioned with its true and
        predicted class (green if the prediction is correct, red otherwise).
        """
        num_images = self.cfg_trainer.log_images
        examples = jax.tree.map(lambda x: x[:num_images], batch)
        outputs = self.predict_step_fn(params, examples)
        examples, outputs = jax.device_get((examples, outputs))

        labels = examples['label'].tolist()
        preds = outputs['logits'].argmax(axis=-1).tolist()
        tiles = [
            caption_image(img, f'true {label} | pred {pred}', 'green' if label == pred else 'red')
            for img, label, pred in zip(examples['img'], labels, preds, strict=True)
        ]
        grid = make_grid(tiles, nrow=8, padding=2, pad_value=255)
        self.writer.add_image('predictions', grid)
