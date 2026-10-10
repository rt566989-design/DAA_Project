# core/__init__.py
"""Core domain models, evaluation engine, and dataset generators."""
from .models import ShiftType, Employee, Schedule, ConstraintRules
from .evaluator import ScheduleEvaluator, EvaluationResult
from .data_generator import ScenarioGenerator
from .shift_manager import ShiftChangeManager, ReplacementCandidate, ShiftOperationResult

__all__ = [
    "ShiftType",
    "Employee",
    "Schedule",
    "ConstraintRules",
    "ScheduleEvaluator",
    "EvaluationResult",
    "ScenarioGenerator",
    "ShiftChangeManager",
    "ReplacementCandidate",
    "ShiftOperationResult",
]

