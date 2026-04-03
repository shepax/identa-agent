import json
import pathlib
from identa.tasks.schema import AlignmentTask, TaskInstance

def load_builtin_task(task_id: str) -> AlignmentTask:
    """Load a specific alignment task from the builtin library."""
    builtin_path = pathlib.Path(__file__).parent / "builtin" / f"{task_id}.json"
    if not builtin_path.exists():
        raise ValueError(f"Task {task_id} not found in builtin tasks.")
        
    data = json.loads(builtin_path.read_text())
    instances = [TaskInstance(**d) for d in data]
    
    return AlignmentTask(
        task_id=task_id,
        name=task_id.replace("_", " ").title(),
        domain="coding",
        description=f"Calibration using {task_id}",
        instances=instances,
        evaluation_metric="text_similarity",
        source="builtin",
        default_prompt="You are a helpful coding assistant. Answer the user's question."
    )

def load_all_builtin_tasks() -> list[AlignmentTask]:
    """Load all available builtin tasks."""
    builtin_dir = pathlib.Path(__file__).parent / "builtin"
    task_ids = [f.stem for f in builtin_dir.glob("*.json")]
    return [load_builtin_task(tid) for tid in task_ids]
