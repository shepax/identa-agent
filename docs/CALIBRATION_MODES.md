# Identa Dual-Mode Calibration

Identa supports two modes for prompt calibration: **Static** and **Agent**. This allows you to balance cost, speed, and quality based on your needs.

## 1. Static Mode (Fast & Free)

Static mode uses a pre-curated library of high-quality questions for specific domains. 

- **Token Cost**: Zero for question generation.
- **Speed**: Instant setup.
- **Best For**: Common domains (Coding, Marketing) where a fixed set of tests is sufficient.

### Usage
```bash
identa calibrate --domain software_developer --static --target-model gpt-4o
```

## 2. Agent Mode (Dynamic & High Quality)

Agent mode uses a "Generator LLM" to create unique, domain-specific questions on the fly.

- **Token Cost**: Small cost for generating questions (~$0.05 - $0.15).
- **Speed**: Adds ~10 seconds for generation.
- **Best For**: Specialized domains or when you need fresh, non-deterministic evaluation data.

### Usage
```bash
identa calibrate --domain marketing_expert --use-agent --target-model claude-3-5-sonnet
```

---

## Domain Catalog

You can explore the built-in domains using the CLI:

### List Domains
```bash
identa calibrate domains
```

### Inspect a Domain
View the static questions and generation prompt for a specific domain:
```bash
identa calibrate inspect software_developer
```

## Available Built-in Domains

| ID | Name | Description |
|---|---|---|
| `software_developer` | Software Developer | Coding, algorithms, and system architecture. |
| `marketing_expert` | Marketing Expert | Copywriting, brand strategy, and campaign planning. |
| `business_analyst` | Business Analyst | Strategy, finance, and reporting. |
| `general_assistant` | General Assistant | Everyday productivity and general knowledge. |

## Adding Custom Domains

To add a custom domain, define it in `src/identa/tasks/domains.py` following the `DomainCalibrationSet` schema.
