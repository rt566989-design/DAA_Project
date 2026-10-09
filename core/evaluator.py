"""
Evaluation engine for shift schedules.
Validates hard constraints, computes preference satisfaction, and calculates Jain's Fairness Index.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from .models import Schedule, Employee, ConstraintRules, ShiftType


@dataclass
class EvaluationResult:
    """Comprehensive evaluation metrics for a generated schedule."""
    is_valid: bool
    hard_violations_count: int
    hard_violations: List[str]
    total_shifts_assigned: int
    required_shifts_total: int
    coverage_percentage: float
    total_preference_score: int
    max_possible_preference: int
    preference_satisfaction_pct: float
    workload_distribution: Dict[str, int]
    workload_mean: float
    workload_std_dev: float
    jains_fairness_index: float
    composite_fitness_score: float  # Higher is better


class ScheduleEvaluator:
    """Evaluates shift schedules against hard constraints and soft multi-objective criteria."""

    @staticmethod
    def evaluate(
        schedule: Schedule,
        employees: List[Employee],
        shift_requirements: Dict[str, int],
        rules: ConstraintRules
    ) -> EvaluationResult:
        hard_violations: List[str] = []
        employees_map = {e.id: e for e in employees}
        n_employees = len(employees)

        # 1. Check Shift Coverage
        total_required = 0
        total_assigned = 0
        for d in range(schedule.days):
            for s in schedule.shifts:
                req = shift_requirements.get(s, 1)
                total_required += req
                assigned = schedule.get_assigned(d, s)
                total_assigned += len(assigned)
                if len(assigned) < req:
                    hard_violations.append(
                        f"Coverage deficit on Day {d+1}, Shift '{s}': Required {req}, but got {len(assigned)}."
                    )
                # Check for duplicate assignment within same shift
                if len(assigned) != len(set(assigned)):
                    hard_violations.append(
                        f"Duplicate assignment of same employee on Day {d+1}, Shift '{s}'."
                    )

        # 2. Check Daily Limit (Max 1 shift per employee per day)
        for d in range(schedule.days):
            daily_assigned_counts: Dict[int, int] = {}
            for s in schedule.shifts:
                for eid in schedule.get_assigned(d, s):
                    daily_assigned_counts[eid] = daily_assigned_counts.get(eid, 0) + 1
            for eid, count in daily_assigned_counts.items():
                if count > rules.max_shifts_per_day:
                    emp_name = employees_map[eid].name if eid in employees_map else f"Emp #{eid}"
                    hard_violations.append(
                        f"Double booking: {emp_name} is assigned {count} shifts on Day {d+1} (Max: {rules.max_shifts_per_day})."
                    )

        # 3. Check Rest Period (Night shift -> Morning shift next day)
        if rules.min_rest_night_to_morning:
            night_shift = ShiftType.NIGHT.value
            morning_shift = ShiftType.MORNING.value
            for d in range(schedule.days - 1):
                night_workers = schedule.get_assigned(d, night_shift)
                morning_workers_next = schedule.get_assigned(d + 1, morning_shift)
                conflicts = set(night_workers).intersection(morning_workers_next)
                for eid in conflicts:
                    emp_name = employees_map[eid].name if eid in employees_map else f"Emp #{eid}"
                    hard_violations.append(
                        f"Ergonomic rest violation: {emp_name} worked Night on Day {d+1} and Morning on Day {d+2}."
                    )

        # 4. Check Maximum Shifts per Employee
        shift_counts = schedule.get_all_shift_counts(employees)
        for emp in employees:
            assigned_count = shift_counts.get(emp.id, 0)
            if assigned_count > emp.max_shifts_per_week:
                hard_violations.append(
                    f"Overtime violation: {emp.name} assigned {assigned_count} shifts (Limit: {emp.max_shifts_per_week})."
                )

        # 5. Check Unavailability
        if rules.respect_unavailability:
            for emp in employees:
                for unav_day in emp.unavailable_days:
                    if 0 <= unav_day < schedule.days:
                        if schedule.is_employee_working_day(emp.id, unav_day):
                            hard_violations.append(
                                f"Unavailability conflict: {emp.name} is on leave on Day {unav_day+1} but was assigned a shift."
                            )

        # Soft Constraint 1: Preferences
        total_pref_score = 0
        for d in range(schedule.days):
            for s in schedule.shifts:
                for eid in schedule.get_assigned(d, s):
                    emp = employees_map.get(eid)
                    if emp:
                        total_pref_score += emp.get_preference(s)

        max_possible_pref = total_assigned * 5 if total_assigned > 0 else 1
        pref_pct = (total_pref_score / max_possible_pref) * 100.0 if total_assigned > 0 else 0.0

        # Soft Constraint 2: Fairness Metrics (Jain's Fairness Index & Workload Variance)
        counts = [shift_counts.get(emp.id, 0) for emp in employees]
        mean_workload = float(np.mean(counts)) if counts else 0.0
        std_workload = float(np.std(counts)) if counts else 0.0

        # Jain's Fairness Index: (sum(x_i))^2 / (n * sum(x_i^2))
        sum_x = sum(counts)
        sum_sq_x = sum(x * x for x in counts)
        if sum_sq_x > 0 and n_employees > 0:
            jains_index = float((sum_x ** 2) / (n_employees * sum_sq_x))
        else:
            jains_index = 1.0

        coverage_pct = (total_assigned / total_required * 100.0) if total_required > 0 else 100.0

        # Composite Fitness Function (Higher is better)
        # Heavily penalizes hard constraint violations
        hard_penalty = len(hard_violations) * 200.0
        soft_reward = pref_pct * 2.0 + jains_index * 100.0 - (std_workload * 15.0)
        composite_fitness = max(0.0, 500.0 - hard_penalty + soft_reward)

        workload_dist = {emp.name: shift_counts.get(emp.id, 0) for emp in employees}

        return EvaluationResult(
            is_valid=(len(hard_violations) == 0),
            hard_violations_count=len(hard_violations),
            hard_violations=hard_violations,
            total_shifts_assigned=total_assigned,
            required_shifts_total=total_required,
            coverage_percentage=round(coverage_pct, 2),
            total_preference_score=total_pref_score,
            max_possible_preference=max_possible_pref,
            preference_satisfaction_pct=round(pref_pct, 2),
            workload_distribution=workload_dist,
            workload_mean=round(mean_workload, 2),
            workload_std_dev=round(std_workload, 2),
            jains_fairness_index=round(jains_index, 4),
            composite_fitness_score=round(composite_fitness, 2)
        )
