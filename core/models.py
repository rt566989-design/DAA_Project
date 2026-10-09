"""
Data models for the Employee Shift Scheduling Problem.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple
import pandas as pd


class ShiftType(str, Enum):
    MORNING = "Morning"
    EVENING = "Evening"
    NIGHT = "Night"

    @classmethod
    def all_shifts(cls) -> List[str]:
        return [cls.MORNING.value, cls.EVENING.value, cls.NIGHT.value]


@dataclass
class Employee:
    """Represents an employee available for shift scheduling."""
    id: int
    name: str
    role: str = "Staff"
    max_shifts_per_week: int = 5
    preferences: Dict[str, int] = field(default_factory=lambda: {
        ShiftType.MORNING.value: 3,
        ShiftType.EVENING.value: 3,
        ShiftType.NIGHT.value: 2
    })  # Scale: 1 (strongly dislike) to 5 (strongly prefer)
    unavailable_days: Set[int] = field(default_factory=set)  # 0-indexed days (e.g., {2, 5})

    def get_preference(self, shift: str) -> int:
        return self.preferences.get(shift, 3)

    def is_available(self, day: int) -> bool:
        return day not in self.unavailable_days


@dataclass
class ConstraintRules:
    """Hard and soft rules for shift scheduling."""
    max_shifts_per_day: int = 1
    max_shifts_per_week: int = 5
    min_rest_night_to_morning: bool = True  # Forbidden to work Morning if worked Night the day before
    respect_unavailability: bool = True
    enforce_fairness: bool = True


class Schedule:
    """Represents a complete or partial shift schedule."""
    def __init__(self, days: int, shifts: Optional[List[str]] = None):
        self.days = days
        self.shifts = shifts if shifts is not None else ShiftType.all_shifts()
        # assignments: (day, shift) -> list of employee IDs
        self.assignments: Dict[Tuple[int, str], List[int]] = {
            (d, s): [] for d in range(self.days) for s in self.shifts
        }

    def assign(self, day: int, shift: str, employee_id: int) -> None:
        if employee_id not in self.assignments[(day, shift)]:
            self.assignments[(day, shift)].append(employee_id)

    def unassign(self, day: int, shift: str, employee_id: int) -> None:
        if employee_id in self.assignments[(day, shift)]:
            self.assignments[(day, shift)].remove(employee_id)

    def get_assigned(self, day: int, shift: str) -> List[int]:
        return self.assignments.get((day, shift), [])

    def get_employee_shift_for_day(self, employee_id: int, day: int) -> Optional[str]:
        for shift in self.shifts:
            if employee_id in self.assignments.get((day, shift), []):
                return shift
        return None

    def is_employee_working_day(self, employee_id: int, day: int) -> bool:
        return self.get_employee_shift_for_day(employee_id, day) is not None

    def get_employee_total_shifts(self, employee_id: int) -> int:
        count = 0
        for d in range(self.days):
            for s in self.shifts:
                if employee_id in self.assignments.get((d, s), []):
                    count += 1
        return count

    def get_all_shift_counts(self, employees: List[Employee]) -> Dict[int, int]:
        return {emp.id: self.get_employee_total_shifts(emp.id) for emp in employees}

    def copy(self) -> "Schedule":
        new_sched = Schedule(self.days, list(self.shifts))
        for key, val in self.assignments.items():
            new_sched.assignments[key] = list(val)
        return new_sched

    def to_dataframe(self, employees_map: Dict[int, Employee], day_labels: Optional[List[str]] = None) -> pd.DataFrame:
        """Converts schedule into a visually intuitive matrix: Rows = Shifts, Columns = Days."""
        if day_labels is None:
            day_labels = [f"Day {d+1}" for d in range(self.days)]
        
        matrix = {day_label: [] for day_label in day_labels}
        for d in range(self.days):
            col_name = day_labels[d]
            for s in self.shifts:
                emp_ids = self.assignments.get((d, s), [])
                names = [employees_map[eid].name for eid in emp_ids if eid in employees_map]
                matrix[col_name].append(", ".join(names) if names else "—")
        
        df = pd.DataFrame(matrix, index=self.shifts)
        df.index.name = "Shift"
        return df

    def to_employee_roster(self, employees: List[Employee], day_labels: Optional[List[str]] = None) -> pd.DataFrame:
        """Converts schedule into an Employee-centric roster: Rows = Employees, Columns = Days."""
        if day_labels is None:
            day_labels = [f"Day {d+1}" for d in range(self.days)]
        
        data = []
        for emp in employees:
            row = {"ID": emp.id, "Employee": emp.name}
            total_shifts = 0
            for d in range(self.days):
                shift = self.get_employee_shift_for_day(emp.id, d)
                row[day_labels[d]] = shift if shift else "OFF"
                if shift:
                    total_shifts += 1
            row["Total Shifts"] = total_shifts
            data.append(row)
        
        return pd.DataFrame(data)
