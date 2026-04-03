import json
from identa.parser.base import *
from pathlib import Path
import hashlib

def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]

class JsonMessagesParser:
    """Parse OpenAI-compatible messages arrays.

    Handles formats:
    1. Array of messages: [{"role": "system", "content": "..."}]
    2. Object with "messages" key: {"messages": [...], "model": "..."}
    3. Object with "system" + "messages": {"system": "...", "messages": [...]}
    """

    def can_parse(self, content: str, path: str | None = None) -> bool:
        try:
            data = json.loads(content.strip())
            if isinstance(data, list):
                return all("role" in m and "content" in m for m in data)
            if isinstance(data, dict):
                return "messages" in data or "system" in data
            return False
        except (json.JSONDecodeError, TypeError):
            return False

    def parse(self, content: str, path: str | None = None) -> PromptTemplate:
        data = json.loads(content.strip())

        messages = []
        system_prompt = None

        if isinstance(data, list):
            raw_messages = data
        else:
            raw_messages = data.get("messages", [])
            if "system" in data:
                system_prompt = data["system"]

        for msg in raw_messages:
            role = MessageRole(msg["role"])
            if role == MessageRole.SYSTEM and system_prompt is None:
                system_prompt = msg["content"]
            else:
                messages.append(Message(
                    role=role,
                    content=msg["content"],
                    name=msg.get("name"),
                    metadata={k: v for k, v in msg.items()
                              if k not in ("role", "content", "name")}
                ))

        # Extract template variables like {task}, {input}, etc.
        all_text = (system_prompt or "") + " ".join(m.content for m in messages)
        import re
        variables = list(set(re.findall(r'\{(\w+)\}', all_text)))

        return PromptTemplate(
            id=_hash_content(content),
            name=Path(path).stem if path else "prompt",
            source_format=PromptFormat.JSON_MESSAGES,
            source_path=path,
            system_prompt=system_prompt,
            messages=messages,
            template_variables=variables,
            tool_definitions=data.get("tools", []) if isinstance(data, dict) else [],
        )

    def reconstruct(self, template: PromptTemplate) -> str:
        """Rebuild JSON messages from PromptTemplate."""
        output = {"messages": template.to_messages_array()}
        if template.tool_definitions:
            output["tools"] = template.tool_definitions
        return json.dumps(output, indent=2)
