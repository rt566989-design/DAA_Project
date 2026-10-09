"""
Greedy Algorithm for Employee Shift Scheduling.
Paradigm: Greedy Choice Property with Priority-based Multi-Factor Heuristic.
Time Complexity: O(D * S * N log N)
Space Complexity: O(D * S + N)
"""
import time
from typing import List, Dict, Any, Tuple
from core.models import Employee, Schedule, ConstraintRules, ShiftType


class GreedyScheduler:
    """
    Greedy Scheduler with fairness balancing and preference maximization.
    Makes locally optimal choices at each shift slot by prioritizing:
    1. Feasibility (avoiding hard constraint conflicts)
    2. Workload balance (giving priority to employees with fewer assigned shifts)
    3. Employee preference satisfaction
    """

    def __init__(self, pref_weight: float = 1.0, workload_weight: float = 2.0):
        self.name = "Greedy Heuristic"
        self.pref_weight = pref_weight
        self.workload_weight = workload_weight

    def solve(
        self,
        employees: List[Employee],
        days: int,
        shift_requirements: Dict[str, int],
        rules: ConstraintRules
    ) -> Tuple[Schedule, Dict[str, Any]]:
        start_time = time.perf_counter()
        schedule = Schedule(days=days)
        shifts_worked: Dict[int, int] = {e.id: 0 for e in employees}
        emp_map = {e.id: e for e in employees}

        # Iterate chronologically day by day, shift by shift
        for d in range(days):
            for s in schedule.shifts:
                req = shift_requirements.get(s, 1)
                assigned_count = 0

                # Determine eligible candidates satisfying hard constraints
                candidates = []
                for emp in employees:
                    eid = emp.id
                    # 1. Unavailability check
                    if rules.respect_unavailability and (d in emp.unavailable_days):
                        continue

                    # 2. Maximum shifts per week limit
                    if shifts_worked[eid] >= emp.max_shifts_per_week:
                        continue

                    # 3. Daily limit (already assigned today)
                    if schedule.is_employee_working_day(eid, d):
                        continue

                    # 4. Ergonomic Rest check (Night yesterday -> Morning today)
                    if rules.min_rest_night_to_morning and s == ShiftType.MORNING.value and d > 0:
                        if eid in schedule.get_assigned(d - 1, ShiftType.NIGHT.value):
                            continue

                    # Heuristic Score: higher preference is good, lower shifts worked is good
                    pref = emp.get_preference(s)
                    load = shifts_worked[eid]
                    # Score calculation: prefer lower current load, higher preference
                    score = (self.pref_weight * pref) - (self.workload_weight * load)
                    candidates.append((score, emp))

                # Sort candidates by heuristic score descending
                candidates.sort(key=lambda item: item[0], reverse=True)

                # Assign top `req` candidates
                for _, emp in candidates[:req]:
                    schedule.assign(d, s, emp.id)
                    shifts_worked[emp.id] += 1
                    assigned_count += 1

                # If requirements couldn't be fulfilled cleanly, attempt soft relaxation for coverage
                if assigned_count < req:
                    deficit = req - assigned_count
                    backup_candidates = []
                    for emp in employees:
                        eid = emp.id
                        # Absolute hard barrier: cannot work two shifts on same day
                        if schedule.is_employee_working_day(eid, d):
                            continue
                        backup_candidates.append((shifts_worked[eid], emp))
                    
                    backup_candidates.sort(key=lambda x: x[0])
                    for _, emp in backup_candidates[:deficit]:
                        schedule.assign(d, s, emp.id)
                        shifts_worked[emp.id] += 1

        elapsed = time.perf_counter() - start_time
        metadata = {
            "algorithm": self.name,
            "execution_time_sec": round(elapsed, 5),
            "time_complexity": "O(D * S * N log N)",
            "space_complexity": "O(D * S + N)",
            "paradigm": "Greedy Choice / Heuristic Search"
        }
        return schedule, metadata
