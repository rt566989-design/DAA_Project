"""
Unit tests to verify algorithmic correctness, constraint enforcement, and robustness.
"""
import unittest
from core.models import Employee, ShiftType, ConstraintRules
from core.evaluator import ScheduleEvaluator
from core.data_generator import ScenarioGenerator
from algorithms.greedy import GreedyScheduler
from algorithms.backtracking import BacktrackingScheduler
from algorithms.branch_and_bound import BranchAndBoundScheduler
from algorithms.genetic import GeneticAlgorithmScheduler


class TestShiftSchedulingAlgorithms(unittest.TestCase):

    def setUp(self):
        # Small dataset suitable for all algorithms (including exact ones)
        self.employees, self.days, self.reqs, self.rules = ScenarioGenerator.get_preset_scenario("small")

    def test_greedy_scheduler(self):
        scheduler = GreedyScheduler()
        sched, meta = scheduler.solve(self.employees, self.days, self.reqs, self.rules)
        eval_res = ScheduleEvaluator.evaluate(sched, self.employees, self.reqs, self.rules)
        self.assertTrue(eval_res.coverage_percentage >= 95.0)
        self.assertIn("execution_time_sec", meta)

    def test_backtracking_scheduler(self):
        scheduler = BacktrackingScheduler(time_limit_sec=4.0)
        sched, meta = scheduler.solve(self.employees, self.days, self.reqs, self.rules)
        eval_res = ScheduleEvaluator.evaluate(sched, self.employees, self.reqs, self.rules)
        self.assertTrue(eval_res.is_valid or eval_res.coverage_percentage >= 90.0)
        self.assertGreater(meta["nodes_explored"], 0)

    def test_branch_and_bound_scheduler(self):
        scheduler = BranchAndBoundScheduler(time_limit_sec=3.0)
        sched, meta = scheduler.solve(self.employees, self.days, self.reqs, self.rules)
        eval_res = ScheduleEvaluator.evaluate(sched, self.employees, self.reqs, self.rules)
        self.assertIsNotNone(sched)
        self.assertGreater(meta["nodes_explored"], 0)

    def test_genetic_algorithm_scheduler(self):
        scheduler = GeneticAlgorithmScheduler(population_size=30, generations=25, seed=42)
        sched, meta = scheduler.solve(self.employees, self.days, self.reqs, self.rules)
        eval_res = ScheduleEvaluator.evaluate(sched, self.employees, self.reqs, self.rules)
        self.assertEqual(meta["generations_run"], 25)
        self.assertGreater(eval_res.composite_fitness_score, 0)


if __name__ == "__main__":
    unittest.main()
