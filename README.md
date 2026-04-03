# Identa — Cross-Model Prompt Migration CLI (v2.0.0)

[![PyPI version](https://img.shields.io/pypi/v/identa-agent.svg)](https://pypi.org/project/identa-agent/)
[![Python Version](https://img.shields.io/pypi/pyversions/identa-agent.svg)](https://pypi.org/project/identa-agent/)
[![Downloads](https://static.pepy.tech/badge/identa-agent)](https://pepy.tech/project/identa-agent)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitHub stars](https://img.shields.io/github/stars/srodriguez/identa-agent.svg?style=social&label=Star)](https://github.com/srodriguez/identa-agent)

Identa is a powerful tool for managing, migrating, and calibrating LLM prompts across different model architectures. It implements the research-backed **PromptBridge** methodology and the **MAP-RPE** iterative refinement engine.

## 🚀 Key Features

*   **Dual-Mode Calibration (MAP-RPE)**: Optimize prompts using **Static** (zero-cost) or **Agent** (dynamic LLM-generated) question sets.
*   **PromptBridge Migration**: Automated cross-model transfer using mapping extraction and adapter layers.
*   **Statistical Analysis**: Measure the "Transfer Gap" (drift) between models with standard deviation and risk metrics.
*   **Persistent Storage**: Automatic SQLite-based prompt versioning, cross-model history, and calibration results.
*   **Multi-Provider Support**: Integrated adapters for **OpenAI**, **Anthropic**, **Google**, **OpenRouter**, **Mistral**, and **Ollama**.
*   **Terminal Native**: Rich logging, tables, and progress monitoring for long-running evolutionary runs.

## 🧠 Research Basis
Based on *PromptBridge: Cross-Model Transfer of LLM Prompts* (arXiv:2512.01420v1) — Accenture / UC Santa Cruz.

## 📦 Installation
Install the core package from PyPI:
```bash
pip install identa-agent
```

### Optional Dependencies
Identa supports several model providers via optional extras:
```bash
# For Google Gemini Support
pip install "identa-agent[google]"

# For Mistral Support
pip install "identa-agent[mistral]"

# For LangChain/DSPy Integrations
pip install "identa-agent[langchain,dspy]"
```

## 🛠 Quick Start

1.  **Configure Environment**: Create a `.env` file with your API keys:
    ```bash
    OPENAI_API_KEY=sk-...
    OPENROUTER_API_KEY=sk-or-...
    ```

2.  **Run Calibration**:
    ```bash
    # Optimize a prompt for gpt-4o using the Software Developer domain
    identa -v calibrate run --domain software_developer --target gpt-4o
    ```

3.  **Migrate Prompts**:
    ```bash
    # Transfer a prompt from gpt-4o to claude-3-5-sonnet and save to a file
    identa migrate --input my-coding-prompt.txt --from gpt-4o --to claude-3-5-sonnet -o results/migrated.txt
    ```

4.  **Inspect Store**:
    ```bash
    # List all successful migrations and calibrations
    identa store list
    ```

## ⚙️ Configuration (`identa.toml`)

You can set project-wide defaults to avoid repetitive CLI flags:

```toml
[calibration]
target_model = "gpt-4o"
reflection_model = "openrouter/google/gemini-2.0-flash-001"
calibration_questions = 20
```

## 📖 Documentation
- [Quickstart Guide](docs/quickstart.md)
- [Calibration Modes (Static vs Agent)](docs/CALIBRATION_MODES.md)
- [Project Changelog](CHANGELOG.md)

---
*Created for the research community and prompt engineering teams.*
