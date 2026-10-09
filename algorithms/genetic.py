"""
Genetic Algorithm for Employee Shift Scheduling.
Paradigm: Evolutionary Metaheuristic / Stochastic Global Optimization.
Features:
  - Chromosome representation over shift slots
  - Tournament Selection
  - Uniform Crossover preserving shift slot coverage
  - Mutation with random employee replacement
  - Elitism (preserves best individuals across generations)
Time Complexity: O(G * P * (D * S)) where G=Generations, P=Population size
Space Complexity: O(P * (D * S))
"""
import random
import time
from typing import List, Dict, Any, Tuple
from core.models import Employee, Schedule, ConstraintRules, ShiftType
from core.evaluator import ScheduleEvaluator


class GeneticAlgorithmScheduler:
    """
    Genetic Algorithm Scheduler for large-scale employee shift scheduling.
    Excels when problem scale causes exponential search spaces to blow up.
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 60,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        elitism_count: int = 2,
        seed: int = 42
    ):
        self.name = "Genetic Algorithm (Evolutionary Metaheuristic)"
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.seed = seed

    def solve(
        self,
        employees: List[Employee],
        days: int,
        shift_requirements: Dict[str, int],
        rules: ConstraintRules
    ) -> Tuple[Schedule, Dict[str, Any]]:
        rng = random.Random(self.seed)
        start_time = time.perf_counter()

        # Build slot list: (day, shift, slot_idx)
        slots = []
        for d in range(days):
            for s in ShiftType.all_shifts():
                req = shift_requirements.get(s, 1)
                for slot_idx in range(req):
                    slots.append((d, s, slot_idx))

        num_slots = len(slots)
        emp_ids = [e.id for e in employees]

        # Helper to convert chromosome (List[int]) to Schedule
        def chromosome_to_schedule(chrom: List[int]) -> Schedule:
            sched = Schedule(days=days)
            for (d, s, _), eid in zip(slots, chrom):
                sched.assign(d, s, eid)
            return sched

        # Fitness evaluator
        def calculate_fitness(chrom: List[int]) -> float:
            sched = chromosome_to_schedule(chrom)
            res = ScheduleEvaluator.evaluate(sched, employees, shift_requirements, rules)
            return res.composite_fitness_score

        # 1. Initialize Population
        population: List[List[int]] = []
        for _ in range(self.population_size):
            chrom = [rng.choice(emp_ids) for _ in range(num_slots)]
            population.append(chrom)

        best_overall_chrom = population[0]
        best_overall_fitness = calculate_fitness(best_overall_chrom)
        fitness_history = []

        # 2. Evolution Loop
        for gen in range(self.generations):
            # Evaluate all fitnesses
            fitness_scores = [calculate_fitness(ind) for ind in population]

            # Track best of current generation
            gen_best_idx = max(range(self.population_size), key=lambda i: fitness_scores[i])
            if fitness_scores[gen_best_idx] > best_overall_fitness:
                best_overall_fitness = fitness_scores[gen_best_idx]
                best_overall_chrom = list(population[gen_best_idx])

            fitness_history.append(round(best_overall_fitness, 2))

            # 3. Selection & Elitism
            sorted_indices = sorted(range(self.population_size), key=lambda i: fitness_scores[i], reverse=True)
            new_population: List[List[int]] = [list(population[i]) for i in sorted_indices[:self.elitism_count]]

            # Tournament Selection Helper
            def tournament() -> List[int]:
                selected = rng.sample(range(self.population_size), self.tournament_size)
                best_selected = max(selected, key=lambda i: fitness_scores[i])
                return population[best_selected]

            # 4. Breeding (Crossover & Mutation)
            while len(new_population) < self.population_size:
                parent1 = tournament()
                parent2 = tournament()

                # Uniform Crossover
                child = []
                for gene1, gene2 in zip(parent1, parent2):
                    child.append(gene1 if rng.random() < 0.5 else gene2)

                # Mutation
                for i in range(num_slots):
                    if rng.random() < self.mutation_rate:
                        child[i] = rng.choice(emp_ids)

                new_population.append(child)

            population = new_population

        elapsed = time.perf_counter() - start_time
        final_schedule = chromosome_to_schedule(best_overall_chrom)

        metadata = {
            "algorithm": self.name,
            "execution_time_sec": round(elapsed, 5),
            "generations_run": self.generations,
            "population_size": self.population_size,
            "best_fitness": round(best_overall_fitness, 2),
            "fitness_history": fitness_history,
            "time_complexity": "O(G * P * (D * S))",
            "space_complexity": "O(P * D * S)",
            "paradigm": "Evolutionary Metaheuristic"
        }
        return final_schedule, metadata
