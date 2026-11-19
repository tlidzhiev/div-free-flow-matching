import logging

import hydra
import jax
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lab.logger import setup_logging
from lab.training.inferencer import Inferencer
from lab.utils.init import set_random_seed
from lab.utils.io import get_root


@hydra.main(
    version_base='1.3',
    config_path=str(get_root() / 'src/lab/configs'),
    config_name='inference',
)
def main(config: DictConfig) -> None:
    """
    Run inference: instantiate everything from the config and run Inferencer.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    save_path = get_root() / 'data' / 'saved' / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)
    setup_logging(save_path)
    logger = logging.getLogger('inference')

    logger.info(f'Config:\n{OmegaConf.to_yaml(config, resolve=True)}')
    device = (
        jax.devices()[0]
        if config.inferencer.device == 'auto'
        else jax.devices(config.inferencer.device)[0]
    )
    logger.info(f'Using device: {device}')

    datasets = instantiate(config.dataset)

    with jax.default_device(device):
        model = instantiate(config.model)

        # Without _convert_='all' the metric classes are turned into structured configs
        metrics = instantiate(config.metrics, _convert_='all')
        logger.info(f'Metrics: {metrics}')

        inferencer = Inferencer(
            config=config,
            device=device,
            model=model,
            datasets=datasets,
            metrics=metrics,
            save_path=save_path,
        )
        logger.info(f'Model:\n{inferencer.model.summary({"params": inferencer.params})}')

        logs = inferencer.run_inference()

    for part, part_logs in logs.items():
        logger.info(f'\n{part} metrics:\n{yaml.dump(part_logs)}')


if __name__ == '__main__':
    main()
