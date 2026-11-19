import hydra
import jax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lab.training.trainer import Trainer
from lab.utils.init import set_random_seed, setup_saving_and_logging
from lab.utils.io import get_root


@hydra.main(
    version_base='1.3',
    config_path=str(get_root() / 'src/lab/configs'),
    config_name='train',
)
def main(config: DictConfig) -> None:
    """
    Train the model: instantiate everything from the config and run Trainer.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    """
    set_random_seed(config.trainer.seed)
    logger = setup_saving_and_logging(config)

    project_config = OmegaConf.to_container(config, resolve=True)
    writer = instantiate(config.writer, project_config)

    logger.info(f'Config:\n{OmegaConf.to_yaml(config, resolve=True)}')
    device = (
        jax.devices()[0]
        if config.trainer.device == 'auto'
        else jax.devices(config.trainer.device)[0]
    )
    logger.info(f'Using device: {device}')

    datasets = instantiate(config.dataset)

    with jax.default_device(device):
        key = jax.random.PRNGKey(config.trainer.seed)
        key, init_key = jax.random.split(key)

        model = instantiate(config.model)
        example = datasets['train'][0]
        variables = model.init(init_key, **example, train=False)
        logger.info(f'Model:\n{model.summary(variables)}')

        lr_scheduler = instantiate(config.lr_scheduler)
        optimizer = instantiate(config.optimizer, learning_rate=lr_scheduler)
        logger.info(f'Optimizer: {optimizer}')
        logger.info(f'LR Scheduler: {lr_scheduler}')

        criterion = instantiate(config.criterion)
        logger.info(f'Criterion: {criterion}')

        # Without _convert_='all' the result is a DictConfig, which turns the metric classes
        # (dataclasses) into structured configs with '???' fields instead of keeping the classes
        metrics = instantiate(config.metrics, _convert_='all')
        logger.info(f'Metrics: {metrics}')

        trainer = Trainer(
            config=config,
            device=device,
            key=key,
            datasets=datasets,
            model=model,
            variables=variables,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            writer=writer,
        )
        trainer.train()


if __name__ == '__main__':
    main()
