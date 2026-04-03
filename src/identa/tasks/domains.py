import json
import pathlib
from identa.tasks.schema import DomainCalibrationSet, TaskInstance

# Built-in Domain Catalog
BUILTIN_DOMAINS: dict[str, DomainCalibrationSet] = {
    "software_developer": DomainCalibrationSet(
        domain_id="software_developer",
        name="Software Developer",
        description="Coding, algorithm design, and system architecture tasks.",
        static_questions=[
            TaskInstance(
                question="Write a Python function to reverse a linked list.",
                answer="def reverse(head): ..."
            ),
            TaskInstance(
                question="Explain the difference between a process and a thread.",
                answer="Process has own memory space, thread shares memory."
            ),
            TaskInstance(
                question="Write a SQL query to find second highest salary.",
                answer="SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees);"
            ),
            TaskInstance(
                question="What is the time complexity of searching for an element in a balanced binary search tree?",
                answer="O(log n) because the height of the tree is logarithmic to the number of elements."
            ),
            TaskInstance(
                question="Write a Python decorator that measures the execution time of a function.",
                answer="import time\ndef timer(func):\n  def wrapper(*args, **kwargs):\n    start = time.time()\n    res = func(*args, **kwargs)\n    print(time.time() - start)\n    return res\n  return wrapper"
            ),
            TaskInstance(
                question="How does a distributed hash table (DHT) handle node failures?",
                answer="It uses consistent hashing and replication. If a node fails, its keys are remapped to the next available nearest node."
            ),
            TaskInstance(
                question="Explain the difference between TCP and UDP protocols.",
                answer="TCP is connection-oriented, reliable, and ordered. UDP is connectionless, fast, but does not guarantee delivery or order."
            ),
            TaskInstance(
                question="Write a regular expression to match a valid IPv4 address.",
                answer=r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
            ),
            TaskInstance(
                question="What is a memory leak, and how can it happen in a garbage-collected language like Java?",
                answer="A memory leak is unreleased memory. In Java, it happens when objects are no longer needed but are still referenced (e.g., in a static List), preventing the garbage collector from freeing them."
            ),
            TaskInstance(
                question="How does React's Virtual DOM improve performance?",
                answer="It minimizes slow direct DOM manipulation by keeping a lightweight virtual representation, calculating the diff, and bulk-applying only the necessary changes to the real DOM."
            ),
            TaskInstance(
                question="Explain the concept of Dependency Injection.",
                answer="A design pattern where an object receives its dependencies from an external source rather than creating them itself, promoting loose coupling and testability."
            ),
            TaskInstance(
                question="Write an efficient algorithm to find the maximum sub-array sum (Kadane's algorithm).",
                answer="def max_subarray(nums):\n  curr_max = global_max = nums[0]\n  for i in range(1, len(nums)):\n    curr_max = max(nums[i], curr_max + nums[i])\n    global_max = max(global_max, curr_max)\n  return global_max"
            ),
        ],
        agent_generation_prompt="""Generate a diverse set of technical coding interview questions. 
Include algorithms, data structures, and system design. 
Format as a JSON list of objects with 'question' and 'answer' keys.""",
        tags=["coding", "technical", "engineering"]
    ),
    "marketing_expert": DomainCalibrationSet(
        domain_id="marketing_expert",
        name="Marketing Expert",
        description="Copywriting, campaign planning, and brand strategy.",
        static_questions=[
            TaskInstance(
                question="Write a catchy headline for a new energy drink.",
                answer="Unleash your inner beast."
            ),
            TaskInstance(
                question="Describe the target audience for high-end organic skincare.",
                answer="Affluent, health-conscious women aged 25-45."
            ),
        ],
        agent_generation_prompt="""Generate creative marketing prompts. 
Include ad copy, brand strategy, and social media planning. 
Format as a JSON list of objects with 'question' and 'answer' keys.""",
        tags=["creative", "marketing", "copywriting"]
    ),
    "business_analyst": DomainCalibrationSet(
        domain_id="business_analyst",
        name="Business Analyst",
        description="Financial modeling, report generation, and data insights.",
        static_questions=[
            TaskInstance(
                question="What are the key components of a SWOT analysis?",
                answer="Strengths, Weaknesses, Opportunities, Threats."
            ),
        ],
        agent_generation_prompt="Generate business analysis prompts including strategy, finance, and reporting.",
        tags=["business", "finance", "strategy"]
    ),
    "general_assistant": DomainCalibrationSet(
        domain_id="general_assistant",
        name="General Assistant",
        description="Everyday productivity and knowledge retrieval.",
        static_questions=[
            TaskInstance(
                question="Summarize the importance of exercise for mental health.",
                answer="Exercise releases endorphins and reduces stress..."
            ),
        ],
        agent_generation_prompt="Generate diverse general purpose assistant queries.",
        tags=["productivity", "knowledge", "general"]
    )
}

def _load_domain_questions(domain: DomainCalibrationSet) -> DomainCalibrationSet:
    filepath = pathlib.Path(__file__).parent / "builtin" / "domains" / f"{domain.domain_id}_questions.json"
    if filepath.exists():
        try:
            data = json.loads(filepath.read_text())
            domain.static_questions = [
                TaskInstance(question=item["question"], answer=item["answer"])
                for item in data
            ]
        except Exception:
            pass
    return domain

def get_domain(domain_id: str) -> DomainCalibrationSet:
    if domain_id not in BUILTIN_DOMAINS:
        raise ValueError(f"Domain '{domain_id}' not found in catalog.")
    return _load_domain_questions(BUILTIN_DOMAINS[domain_id])

def list_domains() -> list[DomainCalibrationSet]:
    return [_load_domain_questions(d) for d in BUILTIN_DOMAINS.values()]
