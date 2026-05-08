# Splendor RL

A full-stack system for training, evaluating, and deploying a Splendor-playing AI agent. Combines Gumbel AlphaZero self-play with a web platform where humans and LLMs compete on a unified leaderboard.

## Features

- **GPU Training** — Gumbel AlphaZero self-play with a fully vectorized PyTorch game engine
- **League System** — Persistent pool of checkpoints rated via anchored Bradley-Terry
- **Web Play** — Flask/Lambda server for human and LLM games against trained agents
- **Unified Ratings** — ML bots, heuristic bots, Claude Sonnet, and humans on one leaderboard
- **AWS Deployment** — CDK stack with Lambda (Docker), DynamoDB, S3, CloudFront

## Project Structure

```
agent/          — Training pipeline, neural net, search, evaluation
play/           — Web application and play server
replay/         — Game replay utilities
replay_webapp/  — React/Vite frontend SPA
infra/          — AWS CDK infrastructure
deploy.sh       — One-command deployment script
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- (Optional) AWS credentials for deployment and Bedrock LLM access

## Setup

```bash
# Create virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# For AWS deployment features
pip install -e ".[aws]"
```

## Usage

### Training

```bash
# Quick smoke test (verifies everything works)
splendor-smoke

# Full training run
splendor-train --run-id my_run

# Hyperparameter tuning
splendor-tune
```

### Evaluation

```bash
# Evaluate a checkpoint against the opponent pool
splendor-eval agent/runs/league/ckpt_02516_i3250.pt

# View league ratings
splendor-league
```

### Play Server

```bash
# Start the local play server
splendor-play
```

### Deployment

```bash
# Deploy to AWS (builds frontend, deploys CDK stack, syncs data)
./deploy.sh
```

## Testing

```bash
pytest
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed system design document.
