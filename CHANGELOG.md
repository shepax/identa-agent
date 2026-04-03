# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] - 2026-04-03

### Added
- **Configurable Model Defaults**: Added `target_model` and `reflection_model` to `identa.toml` to allow running calibration and migrations without explicit CLI model flags.
- **Migration Persistence**: All `identa migrate` results are now automatically saved to the SQLite database `migrations` table.
- **Migration Output Flag**: Added `--output / -o` to the `migrate` command to save the adapted prompt directly to a file.
- **Flexible CLI Flags**: Added `-v` (verbose) and `-q` (quiet) support directly to `migrate`, `calibrate run`, and `drift` subcommands for improved DX.

### [2.0.0] - 2026-03-31

### Added
- **Security Overlay**: Migrated all provider API keys to Pydantic `SecretStr` to prevent accidental leakage in logs/serialization.
- **Resilience Engine**: Centralized `with_retries` utility with exponential backoff applied to all providers.
- **Atomic Transactions**: Integrated SQLite transaction safety for prompt and calibration writes.
- **Interactive Onboarding**: New `identa init` command for guided setup and configuration validation.
- **Standardized DX**: Unified CLI flag system (`--from`, `--to`, `--input`) across all commands.
- **Progress Tracking**: Rich tree-based progress visualization for `migrate` and `calibrate` runs.
- **Configuration Discovery**: Recursive `.toml` configuration searching and robust parsing with `tomllib`.

### Changed
- **Rate Limiter Refactor**: Replaced busy-wait spin lock with efficient `asyncio.sleep` to eliminate concurrency bottlenecks.
- **Command Architecture**: Promoted `calibrate run` to an explicit subcommand; renamed legacy flags for POSIX compliance.
- **Data Schema**: Added composite indexes to `CalibrationRecord` and `MigrationRecord` for O(1) lookups.
- **Documentation**: Overhauled `README.md` and `Quickstart.md` to align with V2 syntax and error patterns.

### Fixed
- **Provider Edge Cases**: Resolved multi-line system prompt extraction in Anthropic and `usage_metadata` guards in Google/OpenAI providers.
- **Input Validation**: Added strict length and presence checks for all database-bound prompt identifiers.
- **Island Manager Bounds**: Implemented population archival limits and fixed dead-code cache stub.

## [1.0.0] - 2026-03-27

### Added
- Initial project scaffolding and `pyproject.toml` configuration.
- Typer-based CLI structure with command shells.
- Canonical `PromptTemplate` and `PromptParser` system with auto-detection.
- Functional Model Provider adapters (Anthropic, Google, Ollama, LiteLLM) with asynchronous rate limiting.
- MAP-RPE Calibration Engine with island-based evolutionary logic and behavioral scoring.
- PromptBridge Transfer Engine for automated cross-model prompt adaptation.
- Built-in alignment task datasets (`synthetic_code`, `code_contests`).
- End-to-end wiring for `identa calibrate` and `identa migrate` CLI commands.
- Enhanced `MockProvider` for robust testing of evolutionary prompt refinement.
- OpenRouter provider integration with automatic fallback resolution for any model ID.
- Global `--verbose` (`-v`) debug mode using Rich logging for real-time engine visibility.
- Persistent storage layer using SQLAlchemy and SQLite for prompt versioning and results.
- Statistical enhancements to `DriftAnalyzer` including transfer gap variance and risk levels.
- **Dual-Mode Calibration**: Added support for Static (pre-curated) and Agent (LLM-generated) calibration questions.
- **Calibration Domains**: Built-in catalog for Software, Marketing, Business, and General Assistant personas.
- New CLI subcommands: `identa calibrate domains` and `identa calibrate inspect [ID]`.
- Enhanced `MAPRPEEngine` with `calibrate_from_domain` for simplified domain-targeted runs.

### Fixed
- Migrated Google provider to the new `google-genai` SDK (addressing deprecation warnings).
- Resolved critical bugs in `.env` configuration loading and Provider Registry initialization.
- Fixed CLI argument ordering and subcommand registration issues in Typer app.
- Corrected various `AttributeError` and `ImportError` bugs in the engine and storage modules.
