"""
Backtracking Algorithm for Employee Shift Scheduling.
Paradigm: State-Space Tree Exploration with Constraint Satisfaction Problem (CSP) Heuristics.
Heuristics used:
  - Minimum Remaining Values (MRV) Variable Ordering
  - Forward Checking Domain Pruning
  - Most Preferred / Least Constraining Value Ordering
Time Complexity: Worst-case O(N^(D*S)), Average pruned: O(b^d)
Space Complexity: O(D * S) recursion stack
"""
import time
from typing import List, Dict, Any, Tuple, Optional, Set
from core.models import Employee, Schedule, ConstraintRules, ShiftType


class BacktrackingScheduler:
    """
    Exact constraint-satisfaction backtracking solver with MRV and Forward Checking.
    Guarantees finding a zero-violation feasible schedule if one exists within time budget.
    """

    def __init__(self, time_limit_sec: float = 5.0, max_nodes: int = 50000):
        self.name = "Backtracking (CSP with MRV & Pruning)"
        self.time_limit_sec = time_limit_sec
        self.max_nodes = max_nodes
        self.nodes_explored = 0
        self.backtrack_count = 0

    def solve(
        self,
        employees: List[Employee],
        days: int,
        shift_requirements: Dict[str, int],
        rules: ConstraintRules
    ) -> Tuple[Schedule, Dict[str, Any]]:
        start_time = time.perf_counter()
        self.nodes_explored = 0
        self.backtrack_count = 0

        # Construct slot variables: list of (day, shift, slot_index)
        slots = []
        for d in range(days):
            for s in ShiftType.all_shifts():
                req = shift_requirements.get(s, 1)
                for slot_idx in range(req):
                    slots.append((d, s, slot_idx))

        schedule = Schedule(days=days)
        shifts_worked: Dict[int, int] = {e.id: 0 for e in employees}
        emp_map = {e.id: e for e in employees}

        success = self._backtrack(
            slots_to_fill=slots,
            schedule=schedule,
            shifts_worked=shifts_worked,
            employees=employees,
            emp_map=emp_map,
            rules=rules,
            start_time=start_time
        )

        elapsed = time.perf_counter() - start_time
        metadata = {
            "algorithm": self.name,
            "execution_time_sec": round(elapsed, 5),
            "nodes_explored": self.nodes_explored,
            "backtracks": self.backtrack_count,
            "solved_completely": success,
            "time_complexity": "Worst O(N^(D*S)), Pruned O(b^d)",
            "space_complexity": "O(D * S)",
            "paradigm": "Exhaustive Search with Pruning / CSP"
        }
        return schedule, metadata

    def _get_legal_candidates(
        self,
        day: int,
        shift: str,
        schedule: Schedule,
        shifts_worked: Dict[int, int],
        employees: List[Employee],
        rules: ConstraintRules
    ) -> List[Employee]:
        candidates = []
        assigned_in_this_slot = schedule.get_assigned(day, shift)

        for emp in employees:
            eid = emp.id
            # 1. Not already assigned to this slot
            if eid in assigned_in_this_slot:
                continue
            # 2. Daily limit (not already working another shift today)
            if schedule.is_employee_working_day(eid, day):
                continue
            # 3. Maximum shifts per week
            if shifts_worked[eid] >= emp.max_shifts_per_week:
                continue
            # 4. Unavailability
            if rules.respect_unavailability and (day in emp.unavailable_days):
                continue
            # 5. Rest period: Night yesterday -> Morning today
            if rules.min_rest_night_to_morning and shift == ShiftType.MORNING.value and day > 0:
                if eid in schedule.get_assigned(day - 1, ShiftType.NIGHT.value):
                    continue
            # 6. Rest period: Night today -> Morning tomorrow
            if rules.min_rest_night_to_morning and shift == ShiftType.NIGHT.value and day < schedule.days - 1:
                if eid in schedule.get_assigned(day + 1, ShiftType.MORNING.value):
                    continue

            candidates.append(emp)
        return candidates

    def _backtrack(
        self,
        slots_to_fill: List[Tuple[int, str, int]],
        schedule: Schedule,
        shifts_worked: Dict[int, int],
        employees: List[Employee],
        emp_map: Dict[int, Employee],
        rules: ConstraintRules,
        start_time: float
    ) -> bool:
        if not slots_to_fill:
            return True  # All slots satisfied

        # Check termination constraints
        if time.perf_counter() - start_time > self.time_limit_sec or self.nodes_explored >= self.max_nodes:
            return False

        self.nodes_explored += 1

        # MRV Heuristic: Choose variable with the fewest legal values
        best_slot_idx = -1
        min_legal_count = float("inf")
        best_candidates: List[Employee] = []

        for idx, (d, s, _) in enumerate(slots_to_fill):
            legals = self._get_legal_candidates(d, s, schedule, shifts_worked, employees, rules)
            legal_count = len(legals)

            # Forward checking: if any slot has 0 candidates, prune immediately!
            if legal_count == 0:
                self.backtrack_count += 1
                return False

            if legal_count < min_legal_count:
                min_legal_count = legal_count
                best_slot_idx = idx
                best_candidates = legals
                if legal_count == 1:
                    break  # Cannot get fewer than 1 without zero

        d, s, slot_num = slots_to_fill[best_slot_idx]
        remaining_slots = slots_to_fill[:best_slot_idx] + slots_to_fill[best_slot_idx + 1:]

        # Value Ordering: Sort candidates by (preference desc, shifts_worked asc)
        best_candidates.sort(
            key=lambda emp: (emp.get_preference(s), -shifts_worked[emp.id]),
            reverse=True
        )

        for emp in best_candidates:
            # Assign
            schedule.assign(d, s, emp.id)
            shifts_worked[emp.id] += 1

            # Recurse
            if self._backtrack(remaining_slots, schedule, shifts_worked, employees, emp_map, rules, start_time):
                return True

            # Unassign / Backtrack
            schedule.unassign(d, s, emp.id)
            shifts_worked[emp.id] -= 1
            self.backtrack_count += 1

        return False
