from identa.parser.base import PromptFormat, PromptParser, PromptTemplate, Message, MessageRole
import hashlib

class RawTextParser:
    def can_parse(self, content: str, path: str | None = None) -> bool:
        return True

    def parse(self, content: str, path: str | None = None) -> PromptTemplate:
        return PromptTemplate(
            id=hashlib.sha256(content.encode()).hexdigest()[:16],
            name="raw_prompt",
            source_format=PromptFormat.RAW_TEXT,
            source_path=path,
            messages=[Message(role=MessageRole.USER, content=content)]
        )

    def reconstruct(self, template: PromptTemplate) -> str:
        return "\n\n".join(m.content for m in template.messages)
