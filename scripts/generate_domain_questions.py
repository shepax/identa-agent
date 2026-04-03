import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
from pathlib import Path
from identa.config.loader import load_config
from identa.providers.registry import ProviderRegistry
from identa.tasks.domains import BUILTIN_DOMAINS
from identa.tasks.generator import QuestionGenerator

async def main():
    model = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    target_count = 100
    
    config = load_config()
    registry = ProviderRegistry(config)
    provider, ident = registry.resolve(model)
    generator = QuestionGenerator(provider, ident.model_id)

    out_dir = Path(__file__).parent.parent / "src" / "identa" / "tasks" / "builtin" / "domains"
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain_id, domain in BUILTIN_DOMAINS.items():
        print(f"\nGenerating {target_count} questions for '{domain.name}' using {model}...")
        
        all_questions = []
        # Large context models often truncate JSON when asked for 100 items. 
        # We request in smaller batches of 25 to ensure valid JSON responses.
        batch_size = 25 
        
        while len(all_questions) < target_count:
            remaining = target_count - len(all_questions)
            request_size = min(batch_size, remaining)
            print(f" Requesting batch of {request_size}...")
            
            try:
                # Add a prompt hint for diversity
                base_prompt = domain.agent_generation_prompt
                domain.agent_generation_prompt = base_prompt + "\nIMPORTANT: Ensure these questions are unique and cover highly obscure or advanced subtopics. Do not repeat basic questions."
                
                instances = await generator.generate_questions(domain, count=request_size)
                
                # Restore original to prevent infinite prompt growth
                domain.agent_generation_prompt = base_prompt
                
                if not instances:
                    print("  Failed to generate valid instances in this batch.")
                    continue
                    
                # The generator might fallback to static questions if it fails to parse JSON
                # Check if it just returned the static questions
                if len(instances) <= len(domain.static_questions) and instances[0].question == domain.static_questions[0].question:
                    print("  Generator fell back to static list. Retrying batch...")
                    continue
                    
                new_items = [{"question": i.question, "answer": i.answer} for i in instances]
                all_questions.extend(new_items)
                print(f"  Received {len(instances)}. Total valid: {len(all_questions)}")
                
            except Exception as e:
                print(f"  Error during generation: {e}")
                
        # Deduplicate
        unique_questions = {q["question"]: q for q in all_questions}.values()
        final_list = list(unique_questions)[:target_count]
        
        out_path = out_dir / f"{domain_id}_questions.json"
        out_path.write_text(json.dumps(final_list, indent=2))
        print(f"Saved {len(final_list)} unique questions to {out_path.name}")

if __name__ == "__main__":
    asyncio.run(main())
