from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

class PromptFormat(Enum):
    RAW_TEXT = "raw_text"
    JSON_MESSAGES = "json_messages"
    YAML_TEMPLATE = "yaml_template"
    LANGCHAIN = "langchain"
    DSPY = "dspy"

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    IPYTHON = "ipython"  # Llama-specific role for tool calls

@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str
    name: str | None = None       # For tool messages
    metadata: dict = field(default_factory=dict)

@dataclass
class PromptTemplate:
    """Canonical internal representation of a prompt.

    Every parser must produce this format. The transfer engine
    operates exclusively on PromptTemplate instances.
    """
    id: str                                # Unique identifier (filename or hash)
    name: str                              # Human-readable name
    source_format: PromptFormat            # Original format detected/specified
    source_path: str | None = None         # File path if loaded from disk

    system_prompt: str | None = None       # System-level instructions
    user_template: str | None = None       # User message template with {variables}
    messages: list[Message] = field(default_factory=list)  # Full message sequence
    template_variables: list[str] = field(default_factory=list)  # Extracted {var} names
    tool_definitions: list[dict] = field(default_factory=list)   # Tool/function schemas
    metadata: dict = field(default_factory=dict)

    def to_flat_text(self) -> str:
        """Collapse to single string for transfer engine input.

        The transfer engine operates on flat text prompts.
        Structure metadata is preserved separately and re-applied
        after transfer via the parser's reconstruct() method.
        """
        parts = []
        if self.system_prompt:
            parts.append(f"[SYSTEM]\n{self.system_prompt}")
        if self.user_template:
            parts.append(f"[USER TEMPLATE]\n{self.user_template}")
        for msg in self.messages:
            parts.append(f"[{msg.role.value.upper()}]\n{msg.content}")
        return "\n\n".join(parts) if parts else ""

    def to_messages_array(self) -> list[dict]:
        """Export as OpenAI-compatible messages array."""
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        for msg in self.messages:
            result.append({"role": msg.role.value, "content": msg.content})
        return result


class PromptParser(Protocol):
    """Protocol all parsers must implement."""

    def can_parse(self, content: str, path: str | None = None) -> bool:
        """Return True if this parser can handle the given content."""
        ...

    def parse(self, content: str, path: str | None = None) -> PromptTemplate:
        """Parse content into a PromptTemplate."""
        ...

    def reconstruct(self, template: PromptTemplate) -> str:
        """Convert PromptTemplate back to the original format string."""
        ...
