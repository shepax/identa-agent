import random
from identa.calibration.types import IslandState, PromptCandidate

class IslandManager:
    """Manages K evolutionary islands with periodic migration."""

    def __init__(
        self,
        num_islands: int = 3,
        archive_size: int = 1000,
        migration_interval: int = 50,
        migration_rate: float = 0.1,
    ):
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.islands: list[IslandState] = [
            IslandState(island_id=i, archive_size=archive_size)
            for i in range(num_islands)
        ]
        self.current_island_idx = 0
        self.total_additions = 0
        self.global_best: PromptCandidate | None = None

    def initialize(self, initial_prompt: str) -> None:
        """Seed all islands with the initial prompt."""
        seed = PromptCandidate(prompt_text=initial_prompt)
        for island in self.islands:
            island.add_candidate(seed)
        self.global_best = seed

    def add_to_current_island(self, candidate: PromptCandidate) -> None:
        """Add candidate to the currently active island."""
        self.islands[self.current_island_idx].add_candidate(candidate)
        self.total_additions += 1
        # Round-robin island assignment
        self.current_island_idx = (self.current_island_idx + 1) % self.num_islands

    def select_parent(self) -> PromptCandidate:
        island = self.islands[self.current_island_idx]
        if not island.population:
            return self.global_best

        roll = random.random()
        if roll < 0.1:  # Elite
            return self.global_best or island.best_candidate
        elif roll < 0.3:  # Exploration
            return random.choice(island.population)
        else:  # Exploitation
            sorted_pop = sorted(
                island.population, key=lambda c: c.combined_score, reverse=True
            )
            top_k = max(1, len(sorted_pop) // 5)
            return random.choice(sorted_pop[:top_k])

    def maybe_migrate(self) -> None:
        """Migrate top candidates between islands every migration_interval additions."""
        if self.total_additions % self.migration_interval != 0:
            return
        if self.num_islands < 2:
            return

        num_migrants = max(1, int(
            min(len(isl.population) for isl in self.islands) * self.migration_rate
        ))

        for i in range(self.num_islands):
            source_island = self.islands[i]
            target_island = self.islands[(i + 1) % self.num_islands]
            migrants = sorted(
                source_island.population,
                key=lambda c: c.combined_score,
                reverse=True
            )[:num_migrants]
            for migrant in migrants:
                target_island.add_candidate(migrant)

    def update_global_best(self) -> None:
        """Update global best across all islands."""
        for island in self.islands:
            if island.best_candidate and (
                self.global_best is None or
                island.best_candidate.combined_score > self.global_best.combined_score
            ):
                self.global_best = island.best_candidate
