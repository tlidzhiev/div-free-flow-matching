import jax
import numpy as np

from lab.logger.image import caption_image, make_grid
from lab.model.physics.spectral import relative_divergence, vorticity

from .base import BaseTrainer
from .steps import Params


def _normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalization of an image to [0, 1]."""
    img = img.astype(np.float32)
    return (img - img.min()) / max(float(img.max() - img.min()), 1e-8)


class Trainer(BaseTrainer):
    """
    Trainer for conditional flow matching models of 2D velocity fields.
    """

    def _log_batch(self, params: Params, batch: dict[str, jax.Array], part: str) -> None:
        """
        On evaluation partitions, generate fields for the conditions of the first samples of
        the batch and log a grid of vorticities (real ones on top, generated ones below,
        captioned with the condition) and the relative divergence of the generated fields.
        """
        if part == 'train':
            return

        num_images = min(self.cfg_trainer.log_images, len(batch['x']))
        if num_images == 0:
            return
        examples = jax.tree.map(lambda x: x[:num_images], batch)
        rng = jax.random.fold_in(self.sample_key, self.global_step)
        generated = self.sample_step_fn(params, examples, rng)

        real_w, gen_w, rel_div, cond = jax.device_get(
            (
                vorticity(examples['x']),
                vorticity(generated),
                relative_divergence(generated).mean(),
                examples['cond'],
            )
        )
        self.writer.add_scalar('sample_rel_div', float(rel_div))

        tiles = [
            caption_image(_normalize(w), f'{prefix} c={c:.3g}')
            for prefix, fields in (('real', real_w), ('gen', gen_w))
            for w, c in zip(fields, cond.tolist(), strict=True)
        ]
        grid = make_grid(tiles, nrow=8, padding=2, pad_value=255)
        self.writer.add_image('vorticity', grid)
