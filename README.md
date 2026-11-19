# Research Project Template

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
  <a href="#credits">Credits</a>
</p>

## About

This repository is a template for JAX research projects (Flax, Optax, Grain, Orbax). It provides Hydra-based configs, a trainer with EMA, bfloat16 mixed precision, gradient accumulation, `jax.jit`-compiled steps, Orbax checkpointing, resuming and early stopping, an inferencer, and CometML experiment tracking. MNIST classification with a small MLP is included as a working example.

## Installation

### Requirements
- Python 3.13.18
- TPU/CUDA (optional, for TPU/GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/research-project-template.git
cd research-project-template

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.13.18
source .venv/bin/activate

# Install dependencies via uv
uv sync --all-groups

# Install pre-commit
pre-commit install
```

### CometML Configuration

CometML is used by default for experiment tracking. Create a `.comet.config` file in the project root:

```ini
[comet]
api_key=YOUR_API_KEY
workspace=YOUR_WORKSPACE
project_name=YOUR_PROJECT_NAME
```

The API key is only needed for online logging: `writer.mode` is `offline` by default.

## How To Use

To run a script (training or inference), use the following command:

```bash
uv run <script> -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

- `<script>`: Use `train` for training or `inference` for inference
- `CONFIG_NAME`: A config file from `src/lab/configs` (`train` and `inference` by default)
- `HYDRA_CONFIG_ARGUMENTS`: Optional Hydra overrides

MNIST is downloaded and indexed into `data/mnist` on the first run. Runs are saved to `logs/{project_name}/{run_name}`, with checkpoints in the `checkpoints` subdirectory.

## Examples

Train the baseline model with online logging:

```bash
uv run train writer.run_name=baseline writer.mode=online
```

Train with the best-checkpoint selection and early stopping on the test accuracy:

```bash
uv run train writer.run_name=baseline trainer.monitor="max test_accuracy" trainer.early_stop=3
```

Training runs for `trainer.total_steps` optimizer steps. Validation runs every
`trainer.eval_every_steps`, and periodic checkpoints are written every
`trainer.checkpoint_every_steps`. Each checkpoint is a directory named after its step
(`checkpoints/1000`); the latest, the periodic and the best (by `trainer.monitor`) ones are kept.
Resume from a step of the same run (or `latest`):

```bash
uv run train writer.run_name=baseline trainer.resume_from=1000
```

The config of the model, optimizer and lr scheduler must stay the same when resuming (including
`trainer.total_steps`, which sets the cosine schedule).

Evaluate a checkpoint with the EMA weights and save the predictions to `data/saved/baseline`:

```bash
uv run inference inferencer.from_pretrained=logs/research-project-template/baseline/checkpoints/1000 \
  inferencer.use_ema=true inferencer.save_path=baseline
```

## Credits

The code of this project is a JAX port of the [PyTorch project template](https://github.com/Blinorot/pytorch_project_template).
