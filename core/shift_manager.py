"""
Dynamic Shift Change & Auto-Replacement Engine.
Algorithmic Paradigms:
  - Constraint Satisfaction Validation (Pruning & Legality Checking)
  - Multi-Objective Greedy Choice (Workload Balancing + Preference Maximization)
  - Automated Shift Swap & Emergency Leave Repair
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import copy

from core.models import Schedule, Employee, ConstraintRules, ShiftType
from core.evaluator import ScheduleEvaluator, EvaluationResult


@dataclass
class ReplacementCandidate:
    """Evaluation record of an employee considered for shift replacement."""
    employee_id: int
    employee_name: str
    is_eligible: bool
    ineligibility_reasons: List[str] = field(default_factory=list)
    preference_score: int = 3
    current_shifts: int = 0
    fairness_benefit: float = 0.0
    suitability_score: float = 0.0


@dataclass
class ShiftOperationResult:
    """Result of an automated shift change, swap, or leave repair."""
    success: bool
    operation_type: str  # 'AUTO_REPLACE', 'AUTO_SWAP', 'SHIFT_CHANGE', 'EMERGENCY_LEAVE'
    message: str
    day: int
    shift: Optional[str] = None
    affected_employees: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_before: Optional[EvaluationResult] = None
    evaluation_after: Optional[EvaluationResult] = None
    violations_after: int = 0


class ShiftChangeManager:
    """
    Automated Shift Management System.
    Provides intelligent auto-replacement, shift swaps, and leave reallocation
    while preserving hard constraints and optimizing preference & fairness.
    """

    @staticmethod
    def check_assignment_legality(
        schedule: Schedule,
        day: int,
        shift: str,
        candidate: Employee,
        employees: List[Employee],
        rules: ConstraintRules,
        ignore_current_day_shift_for_emp: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validates if assigning 'candidate' to (day, shift) violates any hard constraints.
        Returns (is_legal, list_of_reasons_if_illegal).
        """
        reasons: List[str] = []
        emp_map = {e.id: e for e in employees}

        # 1. Leave / Unavailability Check
        if rules.respect_unavailability and day in candidate.unavailable_days:
            reasons.append(f"On scheduled leave on Day {day + 1}")

        # 2. Daily limit (Double-booking check)
        current_shift_today = schedule.get_employee_shift_for_day(candidate.id, day)
        if current_shift_today is not None and not ignore_current_day_shift_for_emp:
            if current_shift_today != shift:
                reasons.append(f"Already working {current_shift_today} shift on Day {day + 1}")
            else:
                reasons.append(f"Already assigned to this {shift} shift on Day {day + 1}")

        # 3. Weekly Max Shifts Limit
        total_shifts = schedule.get_employee_total_shifts(candidate.id)
        # If employee already working on this day and swapping, shift count net change is 0
        is_already_working_today = (current_shift_today is not None)
        net_shifts = total_shifts if is_already_working_today else total_shifts + 1
        if net_shifts > candidate.max_shifts_per_week:
            reasons.append(
                f"Exceeds weekly shift cap ({net_shifts}/{candidate.max_shifts_per_week} shifts)"
            )

        # 4. Ergonomic Rest Rule: Forbidden Night -> Morning transition
        if rules.min_rest_night_to_morning:
            # Check A: If candidate is being assigned Morning on Day d, did they work Night on Day d-1?
            if shift == ShiftType.MORNING.value and day > 0:
                prev_shift = schedule.get_employee_shift_for_day(candidate.id, day - 1)
                if prev_shift == ShiftType.NIGHT.value:
                    reasons.append(f"Ergonomic conflict: Worked Night shift on Day {day} (Day before)")

            # Check B: If candidate is being assigned Night on Day d, are they working Morning on Day d+1?
            if shift == ShiftType.NIGHT.value and day < schedule.days - 1:
                next_shift = schedule.get_employee_shift_for_day(candidate.id, day + 1)
                if next_shift == ShiftType.MORNING.value:
                    reasons.append(f"Ergonomic conflict: Scheduled for Morning shift on Day {day + 2} (Next day)")

        is_legal = (len(reasons) == 0)
        return is_legal, reasons

    @staticmethod
    def evaluate_candidates_for_slot(
        schedule: Schedule,
        day: int,
        shift: str,
        current_emp_id: Optional[int],
        employees: List[Employee],
        rules: ConstraintRules
    ) -> List[ReplacementCandidate]:
        """
        Scrutinizes all employees for filling a slot on (day, shift).
        Ranks eligible candidates using multi-objective scoring (Fairness + Preferences).
        """
        results: List[ReplacementCandidate] = []
        emp_counts = schedule.get_all_shift_counts(employees)
        max_possible_weekly = max((e.max_shifts_per_week for e in employees), default=5)

        for emp in employees:
            if current_emp_id is not None and emp.id == current_emp_id:
                continue

            is_legal, reasons = ShiftChangeManager.check_assignment_legality(
                schedule=schedule,
                day=day,
                shift=shift,
                candidate=emp,
                employees=employees,
                rules=rules,
                ignore_current_day_shift_for_emp=False
            )

            pref_score = emp.get_preference(shift)
            curr_shifts = emp_counts.get(emp.id, 0)

            # Fairness benefit: Higher when employee currently has fewer shifts
            fairness_bonus = max(0.0, float(max_possible_weekly - curr_shifts))

            # Composite suitability score:
            # - High preference weight (scale 1-5 => 10-50 pts)
            # - Fairness weight (fewer current shifts => up to 50 pts)
            suitability = (pref_score * 10.0) + (fairness_bonus * 10.0) if is_legal else -100.0

            results.append(ReplacementCandidate(
                employee_id=emp.id,
                employee_name=emp.name,
                is_eligible=is_legal,
                ineligibility_reasons=reasons,
                preference_score=pref_score,
                current_shifts=curr_shifts,
                fairness_benefit=fairness_bonus,
                suitability_score=suitability
            ))

        # Sort: Eligible first, then descending by suitability score
        results.sort(key=lambda c: (c.is_eligible, c.suitability_score), reverse=True)
        return results

    @classmethod
    def auto_replace_employee(
        cls,
        schedule: Schedule,
        day: int,
        shift: str,
        current_emp_id: int,
        employees: List[Employee],
        shift_requirements: Dict[str, int],
        rules: ConstraintRules,
        preferred_replacement_id: Optional[int] = None
    ) -> ShiftOperationResult:
        """
        Automatically finds the best eligible replacement and transfers the shift.
        """
        emp_map = {e.id: e for e in employees}
        current_emp = emp_map.get(current_emp_id)
        if not current_emp:
            return ShiftOperationResult(
                success=False,
                operation_type="AUTO_REPLACE",
                message=f"Employee ID {current_emp_id} not found.",
                day=day,
                shift=shift
            )

        # Check if current_emp is actually assigned
        assigned_list = schedule.get_assigned(day, shift)
        if current_emp_id not in assigned_list:
            return ShiftOperationResult(
                success=False,
                operation_type="AUTO_REPLACE",
                message=f"{current_emp.name} is not assigned to {shift} on Day {day + 1}.",
                day=day,
                shift=shift
            )

        eval_before = ScheduleEvaluator.evaluate(schedule, employees, shift_requirements, rules)

        # Rank candidates
        candidates = cls.evaluate_candidates_for_slot(
            schedule=schedule,
            day=day,
            shift=shift,
            current_emp_id=current_emp_id,
            employees=employees,
            rules=rules
        )

        chosen_candidate: Optional[ReplacementCandidate] = None

        if preferred_replacement_id is not None:
            # Check if preferred is eligible
            for c in candidates:
                if c.employee_id == preferred_replacement_id:
                    if c.is_eligible:
                        chosen_candidate = c
                    else:
                        return ShiftOperationResult(
                            success=False,
                            operation_type="AUTO_REPLACE",
                            message=f"{c.employee_name} is ineligible: {', '.join(c.ineligibility_reasons)}",
                            day=day,
                            shift=shift
                        )
                    break
        else:
            # Automatically pick the best eligible candidate
            eligible_candidates = [c for c in candidates if c.is_eligible]
            if eligible_candidates:
                chosen_candidate = eligible_candidates[0]

        if not chosen_candidate:
            return ShiftOperationResult(
                success=False,
                operation_type="AUTO_REPLACE",
                message=f"No eligible replacement found for {shift} on Day {day + 1} without violating constraints.",
                day=day,
                shift=shift
            )

        # Execute assignment change
        schedule.unassign(day, shift, current_emp_id)
        schedule.assign(day, shift, chosen_candidate.employee_id)

        eval_after = ScheduleEvaluator.evaluate(schedule, employees, shift_requirements, rules)

        return ShiftOperationResult(
            success=True,
            operation_type="AUTO_REPLACE",
            message=(
                f"Successfully replaced {current_emp.name} with {chosen_candidate.employee_name} "
                f"for {shift} on Day {day + 1}. Preference: {chosen_candidate.preference_score}/5 | "
                f"Jain's Fairness: {eval_after.jains_fairness_index:.4f}"
            ),
            day=day,
            shift=shift,
            affected_employees=[
                {"id": current_emp.id, "name": current_emp.name, "action": "Removed from shift"},
                {"id": chosen_candidate.employee_id, "name": chosen_candidate.employee_name, "action": "Assigned to shift"}
            ],
            evaluation_before=eval_before,
            evaluation_after=eval_after,
            violations_after=eval_after.hard_violations_count
        )

    @classmethod
    def auto_swap_shifts(
        cls,
        schedule: Schedule,
        day: int,
        emp1_id: int,
        emp2_id: int,
        employees: List[Employee],
        shift_requirements: Dict[str, int],
        rules: ConstraintRules
    ) -> ShiftOperationResult:
        """
        Verifies and automatically executes a mutual shift swap between two employees on 'day'.
        """
        emp_map = {e.id: e for e in employees}
        e1 = emp_map.get(emp1_id)
        e2 = emp_map.get(emp2_id)

        if not e1 or not e2:
            return ShiftOperationResult(
                success=False,
                operation_type="AUTO_SWAP",
                message="One or both employees do not exist.",
                day=day
            )

        s1 = schedule.get_employee_shift_for_day(emp1_id, day)
        s2 = schedule.get_employee_shift_for_day(emp2_id, day)

        if s1 is None and s2 is None:
            return ShiftOperationResult(
                success=False,
                operation_type="AUTO_SWAP",
                message=f"Neither {e1.name} nor {e2.name} is assigned to any shift on Day {day + 1}.",
                day=day
            )

        if s1 == s2:
            return ShiftOperationResult(
                success=False,
                operation_type="AUTO_SWAP",
                message=f"Both {e1.name} and {e2.name} already have the exact same status ({s1 or 'OFF'}) on Day {day + 1}.",
                day=day
            )

        eval_before = ScheduleEvaluator.evaluate(schedule, employees, shift_requirements, rules)

        # Temporarily create a copy to test feasibility
        test_sched = schedule.copy()
        if s1:
            test_sched.unassign(day, s1, emp1_id)
        if s2:
            test_sched.unassign(day, s2, emp2_id)

        # Validate e1 taking s2
        if s2 is not None:
            legal_e1, reasons_e1 = cls.check_assignment_legality(
                test_sched, day, s2, e1, employees, rules
            )
            if not legal_e1:
                return ShiftOperationResult(
                    success=False,
                    operation_type="AUTO_SWAP",
                    message=f"Swap blocked: {e1.name} cannot work {s2} ({', '.join(reasons_e1)}).",
                    day=day
                )
            test_sched.assign(day, s2, emp1_id)

        # Validate e2 taking s1
        if s1 is not None:
            legal_e2, reasons_e2 = cls.check_assignment_legality(
                test_sched, day, s1, e2, employees, rules
            )
            if not legal_e2:
                return ShiftOperationResult(
                    success=False,
                    operation_type="AUTO_SWAP",
                    message=f"Swap blocked: {e2.name} cannot work {s1} ({', '.join(reasons_e2)}).",
                    day=day
                )
            test_sched.assign(day, s1, emp2_id)

        # Apply to actual schedule
        if s1:
            schedule.unassign(day, s1, emp1_id)
        if s2:
            schedule.unassign(day, s2, emp2_id)
        if s2:
            schedule.assign(day, s2, emp1_id)
        if s1:
            schedule.assign(day, s1, emp2_id)

        eval_after = ScheduleEvaluator.evaluate(schedule, employees, shift_requirements, rules)

        msg = (
            f"Successfully swapped shifts on Day {day + 1}: "
            f"{e1.name} ({s1 or 'OFF'} -> {s2 or 'OFF'}) and "
            f"{e2.name} ({s2 or 'OFF'} -> {s1 or 'OFF'})."
        )

        return ShiftOperationResult(
            success=True,
            operation_type="AUTO_SWAP",
            message=msg,
            day=day,
            affected_employees=[
                {"id": e1.id, "name": e1.name, "old_shift": s1, "new_shift": s2},
                {"id": e2.id, "name": e2.name, "old_shift": s2, "new_shift": s1}
            ],
            evaluation_before=eval_before,
            evaluation_after=eval_after,
            violations_after=eval_after.hard_violations_count
        )

    @classmethod
    def auto_change_shift(
        cls,
        schedule: Schedule,
        day: int,
        emp_id: int,
        target_shift: str,
        employees: List[Employee],
        shift_requirements: Dict[str, int],
        rules: ConstraintRules
    ) -> ShiftOperationResult:
        """
        Automated Shift Change:
        If Employee 'emp_id' wants 'target_shift' on 'day':
          1. If employee already has target_shift, no-op.
          2. Check if employee can legally work target_shift.
          3. If target_shift slot has capacity, move directly.
          4. If target_shift is full, find the best swap partner currently on target_shift
             who can swap to employee's current shift with minimal penalty / highest suitability.
          5. If employee is currently OFF and wants target_shift, auto-swap with employee on target_shift.
        """
        emp_map = {e.id: e for e in employees}
        emp = emp_map.get(emp_id)
        if not emp:
            return ShiftOperationResult(
                success=False,
                operation_type="SHIFT_CHANGE",
                message=f"Employee ID {emp_id} does not exist.",
                day=day,
                shift=target_shift
            )

        current_shift = schedule.get_employee_shift_for_day(emp_id, day)
        if current_shift == target_shift:
            return ShiftOperationResult(
                success=False,
                operation_type="SHIFT_CHANGE",
                message=f"{emp.name} is already assigned to {target_shift} on Day {day + 1}.",
                day=day,
                shift=target_shift
            )

        # Check if emp can work target_shift legally
        # If currently assigned to another shift today, we ignore today's shift for double booking check
        legal_emp, reasons_emp = cls.check_assignment_legality(
            schedule=schedule,
            day=day,
            shift=target_shift,
            candidate=emp,
            employees=employees,
            rules=rules,
            ignore_current_day_shift_for_emp=(current_shift is not None)
        )

        if not legal_emp:
            return ShiftOperationResult(
                success=False,
                operation_type="SHIFT_CHANGE",
                message=f"Cannot move {emp.name} to {target_shift}: {', '.join(reasons_emp)}.",
                day=day,
                shift=target_shift
            )

        required_headcount = shift_requirements.get(target_shift, 1)
        current_target_workers = list(schedule.get_assigned(day, target_shift))

        # Case A: There is free slot capacity in target shift
        if len(current_target_workers) < required_headcount:
            eval_before = ScheduleEvaluator.evaluate(schedule, employees, shift_requirements, rules)
            if current_shift:
                schedule.unassign(day, current_shift, emp_id)
            schedule.assign(day, target_shift, emp_id)
            eval_after = ScheduleEvaluator.evaluate(schedule, employees, shift_requirements, rules)
            return ShiftOperationResult(
                success=True,
                operation_type="SHIFT_CHANGE",
                message=f"Directly moved {emp.name} from {current_shift or 'OFF'} to {target_shift} on Day {day + 1}.",
                day=day,
                shift=target_shift,
                evaluation_before=eval_before,
                evaluation_after=eval_after,
                violations_after=eval_after.hard_violations_count
            )

        # Case B: Target shift is full -> Search for the optimal swap partner in target_shift!
        best_partner_id = None
        best_partner_score = -9999.0
        best_partner_reason = ""

        for candidate_id in current_target_workers:
            c_emp = emp_map[candidate_id]
            # Check if c_emp can legally take emp's current_shift (or OFF if emp was OFF)
            if current_shift is not None:
                legal_c, reasons_c = cls.check_assignment_legality(
                    schedule=schedule,
                    day=day,
                    shift=current_shift,
                    candidate=c_emp,
                    employees=employees,
                    rules=rules,
                    ignore_current_day_shift_for_emp=True
                )
                if not legal_c:
                    continue
                # Preference score of candidate for current_shift
                c_pref = c_emp.get_preference(current_shift)
            else:
                # emp was OFF, so candidate would be taking OFF (weekly shift count drops by 1)
                # Check leave constraint on c_emp (none since they are going OFF)
                c_pref = 3

            # Multiobjective suitability of this swap
            emp_pref_gain = emp.get_preference(target_shift) - (emp.get_preference(current_shift) if current_shift else 3)
            swap_score = (c_pref * 5.0) + (emp_pref_gain * 10.0)

            if swap_score > best_partner_score:
                best_partner_score = swap_score
                best_partner_id = candidate_id

        if best_partner_id is not None:
            # Execute the auto-swap
            return cls.auto_swap_shifts(
                schedule=schedule,
                day=day,
                emp1_id=emp_id,
                emp2_id=best_partner_id,
                employees=employees,
                shift_requirements=shift_requirements,
                rules=rules
            )

        return ShiftOperationResult(
            success=False,
            operation_type="SHIFT_CHANGE",
            message=(
                f"Shift change could not find a feasible swap partner in {target_shift} on Day {day + 1}. "
                "All workers in that shift have constraint conflicts with the swap."
            ),
            day=day,
            shift=target_shift
        )

    @classmethod
    def handle_emergency_leave(
        cls,
        schedule: Schedule,
        day: int,
        emp_id: int,
        employees: List[Employee],
        shift_requirements: Dict[str, int],
        rules: ConstraintRules
    ) -> ShiftOperationResult:
        """
        Handles sudden employee absence or emergency leave request:
        1. Marks day as unavailable for employee.
        2. If assigned to a shift, automatically replaces them with the best eligible candidate.
        """
        emp_map = {e.id: e for e in employees}
        emp = emp_map.get(emp_id)
        if not emp:
            return ShiftOperationResult(
                success=False,
                operation_type="EMERGENCY_LEAVE",
                message=f"Employee ID {emp_id} not found.",
                day=day
            )

        emp.unavailable_days.add(day)
        current_shift = schedule.get_employee_shift_for_day(emp_id, day)

        if not current_shift:
            return ShiftOperationResult(
                success=True,
                operation_type="EMERGENCY_LEAVE",
                message=f"{emp.name} was already OFF on Day {day + 1}. Marked Day {day + 1} as leave.",
                day=day,
                affected_employees=[{"id": emp.id, "name": emp.name, "action": "Leave granted"}]
            )

        # Automatically replace the employee on that shift
        replace_result = cls.auto_replace_employee(
            schedule=schedule,
            day=day,
            shift=current_shift,
            current_emp_id=emp_id,
            employees=employees,
            shift_requirements=shift_requirements,
            rules=rules
        )

        if replace_result.success:
            replace_result.operation_type = "EMERGENCY_LEAVE"
            replace_result.message = (
                f"[Emergency Leave Granted] {emp.name} marked OFF on Day {day + 1}. "
                f"Automatically replaced by {replace_result.affected_employees[1]['name']} on {current_shift} shift."
            )
            return replace_result
        else:
            # Could not find replacement, but still unassign to respect leave
            schedule.unassign(day, current_shift, emp_id)
            eval_after = ScheduleEvaluator.evaluate(schedule, employees, shift_requirements, rules)
            return ShiftOperationResult(
                success=True,
                operation_type="EMERGENCY_LEAVE",
                message=(
                    f"[Notice: Emergency Leave] Granted for {emp.name} on Day {day + 1}. "
                    f"Notice: No feasible replacement was available without overtime/rest conflicts. Slot is currently open."
                ),
                day=day,
                shift=current_shift,
                evaluation_after=eval_after,
                violations_after=eval_after.hard_violations_count
            )
