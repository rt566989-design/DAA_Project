"""
Unit tests for ShiftChangeManager (Automated Shift Change & Swap Engine).
"""
import unittest
from core.models import Employee, ShiftType, ConstraintRules
from core.evaluator import ScheduleEvaluator
from core.data_generator import ScenarioGenerator
from core.shift_manager import ShiftChangeManager
from algorithms.greedy import GreedyScheduler


class TestShiftChangeManager(unittest.TestCase):

    def setUp(self):
        self.employees, self.days, self.reqs, self.rules = ScenarioGenerator.get_preset_scenario("small")
        scheduler = GreedyScheduler()
        self.schedule, _ = scheduler.solve(self.employees, self.days, self.reqs, self.rules)

    def test_auto_replace_employee(self):
        # Pick day 0 and employee assigned to Morning
        morning_assigned = self.schedule.get_assigned(0, ShiftType.MORNING.value)
        self.assertTrue(len(morning_assigned) > 0)
        emp_to_replace = morning_assigned[0]

        result = ShiftChangeManager.auto_replace_employee(
            schedule=self.schedule,
            day=0,
            shift=ShiftType.MORNING.value,
            current_emp_id=emp_to_replace,
            employees=self.employees,
            shift_requirements=self.reqs,
            rules=self.rules
        )

        if result.success:
            self.assertNotIn(emp_to_replace, self.schedule.get_assigned(0, ShiftType.MORNING.value))
            # New assigned employee should be in the shift
            new_emp_id = result.affected_employees[1]["id"]
            self.assertIn(new_emp_id, self.schedule.get_assigned(0, ShiftType.MORNING.value))
            self.assertIsNotNone(result.evaluation_after)

    def test_auto_swap_shifts(self):
        day = 1
        m_assigned = self.schedule.get_assigned(day, ShiftType.MORNING.value)
        e_assigned = self.schedule.get_assigned(day, ShiftType.EVENING.value)

        if m_assigned and e_assigned:
            emp1 = m_assigned[0]
            emp2 = e_assigned[0]

            result = ShiftChangeManager.auto_swap_shifts(
                schedule=self.schedule,
                day=day,
                emp1_id=emp1,
                emp2_id=emp2,
                employees=self.employees,
                shift_requirements=self.reqs,
                rules=self.rules
            )
            # Either successful swap or blocked with a valid constraint reason
            self.assertIsInstance(result.success, bool)
            if result.success:
                self.assertIn(emp2, self.schedule.get_assigned(day, ShiftType.MORNING.value))
                self.assertIn(emp1, self.schedule.get_assigned(day, ShiftType.EVENING.value))

    def test_auto_change_shift(self):
        day = 2
        # Pick an employee working in Evening and change to Night
        e_assigned = self.schedule.get_assigned(day, ShiftType.EVENING.value)
        if e_assigned:
            emp_id = e_assigned[0]
            result = ShiftChangeManager.auto_change_shift(
                schedule=self.schedule,
                day=day,
                emp_id=emp_id,
                target_shift=ShiftType.NIGHT.value,
                employees=self.employees,
                shift_requirements=self.reqs,
                rules=self.rules
            )
            self.assertIsInstance(result.success, bool)

    def test_emergency_leave_auto_handling(self):
        day = 3
        # Pick any employee working on day 3
        working_emp = None
        working_shift = None
        for s in ShiftType.all_shifts():
            assigned = self.schedule.get_assigned(day, s)
            if assigned:
                working_emp = assigned[0]
                working_shift = s
                break

        if working_emp:
            result = ShiftChangeManager.handle_emergency_leave(
                schedule=self.schedule,
                day=day,
                emp_id=working_emp,
                employees=self.employees,
                shift_requirements=self.reqs,
                rules=self.rules
            )
            self.assertTrue(result.success)
            self.assertNotIn(working_emp, self.schedule.get_assigned(day, working_shift))


if __name__ == "__main__":
    unittest.main()
