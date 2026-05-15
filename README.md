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

## Plugin Integration

### Claude

The `.claude-plugin/` directory contains configuration for integrating skills with Claude AI. The `plugin.json` defines the plugin manifest and `marketplace.json` lists all available skills.

### Cursor

The `.cursor-plugin/` directory provides similar integration for the Cursor IDE, enabling AI-assisted coding with custom skills.

## Contributing

1. Fork this repository
2. Create a feature branch: `git checkout -b feat/my-new-skill`
3. Add your skill definition to the appropriate marketplace JSON
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Security

Please review our [Security Policy](.github/workflows/SECURITY.md) before reporting vulnerabilities.

## License

This project is licensed under the Apache 2.0 License — see [LICENSE](LICENSE) for details.
