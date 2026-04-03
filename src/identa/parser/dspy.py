from identa.parser.base import PromptFormat, PromptTemplate

class DSPyParser:
    def can_parse(self, content: str, path: str | None = None) -> bool:
        return False

    def parse(self, content: str, path: str | None = None) -> PromptTemplate:
        raise NotImplementedError()

    def reconstruct(self, template: PromptTemplate) -> str:
        raise NotImplementedError()
