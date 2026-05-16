# Skills

A fork of [huggingface/skills](https://huggingface.co/skills) — a platform for discovering, evaluating, and sharing AI skills/tools that can be used by language model agents.

## Overview

This repository contains:
- **Skills marketplace** — A curated collection of AI skills/tools
- **Agent generation** — Automated workflows for generating agents from skills
- **Evaluation leaderboards** — Track and compare skill performance across models
- **Hacker leaderboards** — Community contributions and rankings

## Structure

```
├── .claude-plugin/          # Claude AI plugin configuration
│   ├── plugin.json          # Plugin metadata and settings
│   └── marketplace.json     # Available skills for Claude
├── .cursor-plugin/          # Cursor IDE plugin configuration
│   ├── plugin.json          # Plugin metadata and settings
│   └── marketplace.json     # Available skills for Cursor
├── .github/
│   └── workflows/
│       ├── generate-agents.yml          # Auto-generate agents from skills
│       ├── push-evals-leaderboard.yml   # Update evals leaderboard
│       └── push-hackers-leaderboard.yml # Update hackers leaderboard
```

## Getting Started

### Prerequisites

- Python 3.9+
- `pip` or `uv` for package management

### Installation

```bash
git clone https://github.com/your-org/skills.git
cd skills
pip install -r requirements.txt
```

### Usage

#### Browse the Marketplace

Skills are defined in the marketplace JSON files and can be loaded programmatically:

```python
import json

with open('.claude-plugin/marketplace.json') as f:
    marketplace = json.load(f)

for skill in marketplace['skills']:
    print(f"{skill['name']}: {skill['description']}")
```

#### Running Evaluations

Evaluations are triggered automatically via GitHub Actions on push to `main`, or can be run manually:

```bash
# Trigger evaluation workflow
gh workflow run push-evals-leaderboard.yml
```

## Local LLM Setup (Ollama + LiteLLM)

This fork is primarily used with local models. To point the tooling at a local Ollama instance instead of hosted APIs, set the following environment variables before running any scripts:

```bash
export OPENAI_API_BASE="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"  # Ollama doesn't require a real key
export DEFAULT_MODEL="ollama/mistral-nemo"  # mistral-nemo works better for me than llama3.1 — snappier responses and solid tool-use
```

Most skills work out of the box with this config. A few that rely on function-calling may need a model with tool-use support (e.g. `llama3.1`, `mistral-nemo`).

## Plugin Integration

### Claude

The `.claude-plugin/` directory contains configuration for integrating skills with Claude AI. The `plugin.json` defines the plugin manifest and `marketplace.json` lists all available skills.

### Cursor

The `.cursor-plugin/` directory provides similar integration for the Cursor IDE, enabling AI-assisted coding with custom skills.

## Contributing

1. Fork this repository
2. Create a feature branch: `git checkout -b feat/my-ne
