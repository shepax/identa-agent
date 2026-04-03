from identa.parser.json_messages import JsonMessagesParser
from identa.parser.raw_text import RawTextParser
from identa.parser.base import MessageRole

def test_json_messages_parser():
    parser = JsonMessagesParser()
    content = '[{"role": "system", "content": "You are a bot"}, {"role": "user", "content": "Hi"}]'
    template = parser.parse(content)
    
    assert template.system_prompt == "You are a bot"
    assert len(template.messages) == 1
    assert template.messages[0].role == MessageRole.USER
    assert template.messages[0].content == "Hi"

def test_raw_text_parser():
    parser = RawTextParser()
    content = "Just a simple prompt"
    template = parser.parse(content)
    
    assert len(template.messages) == 1
    assert template.messages[0].content == "Just a simple prompt"
