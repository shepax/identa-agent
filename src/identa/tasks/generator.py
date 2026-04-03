import json
import logging
from identa.tasks.schema import TaskInstance, DomainCalibrationSet
from identa.providers.base import ModelProvider, CompletionRequest

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """Uses an LLM to generate dynamic, domain-specific calibration questions."""

    def __init__(self, provider: ModelProvider, model_id: str):
        self.provider = provider
        self.model_id = model_id

    async def generate_questions(
        self, 
        domain: DomainCalibrationSet, 
        count: int = 5
    ) -> list[TaskInstance]:
        """Generate a custom set of question-answer pairs for a domain."""
        prompt = f"{domain.agent_generation_prompt}\n\nGenerate exactly {count} unique pairs."
        
        logger.info(f"Generating {count} questions for domain '{domain.domain_id}' using {self.model_id}")
        
        request = CompletionRequest(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        response = await self.provider.complete(request)
        
        try:
            # Attempt to parse JSON from the response
            # Note: We might need more robust parsing logic if LLM includes markdown backticks
            content = response.content
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[-1].split("```")[0].strip()
            
            data = json.loads(content)
            instances = [
                TaskInstance(question=item["question"], answer=item["answer"])
                for item in data[:count]
            ]
            return instances
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse generated questions: {e}")
            logger.debug(f"Raw response: {response.content}")
            # Fallback to static if generation fails
            return domain.static_questions[:count]
