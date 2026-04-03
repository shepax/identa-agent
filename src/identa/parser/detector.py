from pathlib import Path
from identa.parser.base import PromptFormat, PromptParser, PromptTemplate
from identa.parser.raw_text import RawTextParser
from identa.parser.json_messages import JsonMessagesParser
from identa.parser.yaml_template import YamlTemplateParser
# LangChain and DSPy parsers imported conditionally

# Detection priority order
PARSERS: list[tuple[PromptFormat, type[PromptParser]]] = [
    (PromptFormat.JSON_MESSAGES, JsonMessagesParser),
    (PromptFormat.YAML_TEMPLATE, YamlTemplateParser),
    (PromptFormat.RAW_TEXT, RawTextParser),  # Fallback — always succeeds
]

def _get_parser(fmt: PromptFormat) -> PromptParser:
    for f, p in PARSERS:
        if f == fmt:
            return p()
    raise ValueError(f"No parser available for format: {fmt}")

def detect_and_parse(
    content: str,
    path: str | None = None,
    force_format: PromptFormat | None = None
) -> PromptTemplate:
    """Auto-detect prompt format and parse.

    Detection rules:
    1. If force_format specified, use that parser directly
    2. File extension hints: .json → JSON_MESSAGES, .yaml/.yml → YAML_TEMPLATE
    3. Content sniffing: try JSON parse, try YAML parse, fall back to raw text
    4. Raw text always succeeds as final fallback
    """
    if force_format:
        parser = _get_parser(force_format)
        return parser.parse(content, path)

    # Extension-based hint
    if path:
        ext = Path(path).suffix.lower()
        hint_map = {
            ".json": PromptFormat.JSON_MESSAGES,
            ".yaml": PromptFormat.YAML_TEMPLATE,
            ".yml": PromptFormat.YAML_TEMPLATE,
        }
        if ext in hint_map:
            parser = _get_parser(hint_map[ext])
            if parser.can_parse(content, path):
                return parser.parse(content, path)

    # Content sniffing
    for fmt, parser_cls in PARSERS:
        parser = parser_cls()
        if parser.can_parse(content, path):
            return parser.parse(content, path)

    raise ValueError(f"Could not detect prompt format for: {path or 'input'}")


def parse_directory(
    directory: Path,
    force_format: PromptFormat | None = None,
    extensions: set[str] = {".txt", ".md", ".json", ".yaml", ".yml", ".py"}
) -> list[PromptTemplate]:
    """Parse all prompt files in a directory."""
    templates = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() in extensions and file_path.is_file():
            content = file_path.read_text(encoding="utf-8")
            template = detect_and_parse(content, str(file_path), force_format)
            templates.append(template)
    return templates
