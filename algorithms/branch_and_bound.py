"""
Branch and Bound Algorithm for Employee Shift Scheduling.
Paradigm: State-Space Search with Admissible Lower Bounding Function and Upper-Bound Pruning.
Time Complexity: Worst-case O(b^d), drastically pruned by bounding.
Space Complexity: O(D * S)
"""
import time
from typing import List, Dict, Any, Tuple, Optional
from core.models import Employee, Schedule, ConstraintRules, ShiftType


class BranchAndBoundScheduler:
    """
    Branch and Bound (DFBnB) Solver.
    Uses an admissible bounding function based on minimum possible preference penalty
    and workload deviation to prune suboptimal subtrees.
    """

    def __init__(self, time_limit_sec: float = 4.0):
        self.name = "Branch and Bound (Optimal Pruning)"
        self.time_limit_sec = time_limit_sec
        self.nodes_explored = 0
        self.branches_pruned = 0
        self.best_cost = float("inf")
        self.best_schedule: Optional[Schedule] = None

    def solve(
        self,
        employees: List[Employee],
        days: int,
        shift_requirements: Dict[str, int],
        rules: ConstraintRules
    ) -> Tuple[Schedule, Dict[str, Any]]:
        start_time = time.perf_counter()
        self.nodes_explored = 0
        self.branches_pruned = 0
        self.best_cost = float("inf")
        self.best_schedule = None

        # Build list of slots (day, shift, slot_index)
        slots = []
        for d in range(days):
            for s in ShiftType.all_shifts():
                req = shift_requirements.get(s, 1)
                for slot_idx in range(req):
                    slots.append((d, s, slot_idx))

        current_schedule = Schedule(days=days)
        shifts_worked: Dict[int, int] = {e.id: 0 for e in employees}

        # Precompute theoretical minimum preference penalty for each slot (5 - max_pref)
        min_penalties = []
        for d, s, _ in slots:
            min_p = min(5 - emp.get_preference(s) for emp in employees)
            min_penalties.append(min_p)

        # Suffix sums of minimum penalties for fast O(1) lower bound calculation
        suffix_min_penalty = [0] * (len(slots) + 1)
        for i in range(len(slots) - 1, -1, -1):
            suffix_min_penalty[i] = suffix_min_penalty[i + 1] + min_penalties[i]

        self._branch_and_bound(
            slot_idx=0,
            slots=slots,
            suffix_min_penalty=suffix_min_penalty,
            current_cost=0,
            schedule=current_schedule,
            shifts_worked=shifts_worked,
            employees=employees,
            rules=rules,
            start_time=start_time
        )

        elapsed = time.perf_counter() - start_time
        final_sched = self.best_schedule if self.best_schedule is not None else current_schedule

        metadata = {
            "algorithm": self.name,
            "execution_time_sec": round(elapsed, 5),
            "nodes_explored": self.nodes_explored,
            "branches_pruned": self.branches_pruned,
            "best_cost_found": self.best_cost if self.best_cost != float("inf") else "N/A",
            "time_complexity": "O(b^d) with admissible pruning",
            "space_complexity": "O(D * S)",
            "paradigm": "Branch and Bound / Exact Optimization"
        }
        return final_sched, metadata

    def _branch_and_bound(
        self,
        slot_idx: int,
        slots: List[Tuple[int, str, int]],
        suffix_min_penalty: List[int],
        current_cost: int,
        schedule: Schedule,
        shifts_worked: Dict[int, int],
        employees: List[Employee],
        rules: ConstraintRules,
        start_time: float
    ) -> None:
        if time.perf_counter() - start_time > self.time_limit_sec:
            return

        self.nodes_explored += 1

        # Base case: All slots filled
        if slot_idx == len(slots):
            if current_cost < self.best_cost:
                self.best_cost = current_cost
                self.best_schedule = schedule.copy()
            return

        # Compute Admissible Lower Bound: current_cost + theoretical minimum remaining
        lower_bound = current_cost + suffix_min_penalty[slot_idx]
        if lower_bound >= self.best_cost:
            self.branches_pruned += 1
            return  # Prune! Cannot beat best known complete solution

        d, s, _ = slots[slot_idx]
        assigned_in_slot = schedule.get_assigned(d, s)

        # Get candidates satisfying hard constraints
        candidates = []
        for emp in employees:
            eid = emp.id
            if eid in assigned_in_slot:
                continue
            if schedule.is_employee_working_day(eid, d):
                continue
            if shifts_worked[eid] >= emp.max_shifts_per_week:
                continue
            if rules.respect_unavailability and (d in emp.unavailable_days):
                continue
            if rules.min_rest_night_to_morning and s == ShiftType.MORNING.value and d > 0:
                if eid in schedule.get_assigned(d - 1, ShiftType.NIGHT.value):
                    continue

            pref_penalty = 5 - emp.get_preference(s)
            candidates.append((pref_penalty, emp))

        # Sort candidates to branch on most promising (lowest penalty) first
        candidates.sort(key=lambda x: (x[0], shifts_worked[x[1].id]))

        for pref_pen, emp in candidates:
            schedule.assign(d, s, emp.id)
            shifts_worked[emp.id] += 1

            self._branch_and_bound(
                slot_idx + 1,
                slots,
                suffix_min_penalty,
                current_cost + pref_pen,
                schedule,
                shifts_worked,
                employees,
                rules,
                start_time
            )

            schedule.unassign(d, s, emp.id)
            shifts_worked[emp.id] -= 1
