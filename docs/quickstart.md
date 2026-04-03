# Identa Quickstart

Identa is a CLI tool for cross-model prompt migration using **PromptBridge** and **MAP-RPE**.

## 1. Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/identa-ia/identa-ia.git
cd identa-ia
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,google,openrouter]"
```

## 2. Configuration

Create a `.env` file in the root directory with your API keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

## 3. Usage Examples

### Calibrate a Prompt (MAP-RPE)
Find the optimal prompt for a specific model:
```bash
# Basic run
identa calibrate run --domain software_developer --target gpt-4o

# Verbose run (recommended for monitoring progress)
identa -v calibrate run --domain software_developer --target openrouter/google/gemini-2.0-flash-001
```

### Migrate a Prompt (PromptBridge)
Transfer a high-performing prompt from a source model to a target model:
```bash
identa migrate \
  --from gpt-4 \
  --to claude-3-5-sonnet \
  --input my-app-prompt.txt
```

### Inspect Stored Prompts
List all prompts saved in the local SQLite store:
```bash
identa store list
```

### Validating a Run with OpenRouter (Step-by-Step)
To validate that the system works correctly from start to finish using OpenRouter, follow these steps:

1. **Check Available Domains:**
   See which calibration contexts are available:
   ```bash
   identa calibrate domains
   ```

2. **Run Calibration:**
   Start the evolutionary engine. Since you are using OpenRouter, target models should use the `openrouter/` prefix. (Note: using `--questions 5` and `--steps 3` for a faster validation run).
   ```bash
   identa calibrate run \
     --domain software_developer \
     --target "openrouter/openai/gpt-4o-mini" \
     --source-prompt "You are an expert python software developer." \
     --questions 5 \
     --steps 3
   ```

3. **Save the Result:**
   Once calibration finishes, save the generated "Optimal prompt" into a plain text file, such as `optimized_prompt.txt`.

4. **Measure the Drift:**
   Verify the performance gap by testing the prompt across models:
   ```bash
   identa drift --from "openrouter/openai/gpt-4o" --to "openrouter/openai/gpt-4o-mini" --input optimized_prompt.txt
   ```

## 4. Key Global Flags

- `-v, --verbose`: Enable debug logging to see LLM thoughts and engine logs.
- `-q, --quiet`: Suppress all output except fatal errors.
- `--config, -c`: Use a custom `identa.toml` configuration file.
- `--version`: Print tools version.

## 5. Common Errors

### ✗ Authentication failed
This happens when Identa cannot authenticate with the model provider.
**Fix**: Ensure your API key is exported as an environment variable (e.g., `export OPENAI_API_KEY=sk-...`) or run `identa init` again to generate a new `identa.toml` config file wrapper.

### ⚠ Rate limited
This happens when your provider enforces rate limits that Identa breached, or when you run locally and your GPU resources max out.
**Fix**: The tool will automatically retry with exponential backoff. If it still fails, increase the limits on the provider's dashboard or lower your calibration iterations.

### ✗ Input not found
This happens when you provide a file that doesn't exist during `identa migrate --input <file>`.
**Fix**: Verify your path with `ls -la <file>`. Ensure you're specifying the file extension (e.g. `main.py` or `prompt.txt`).
