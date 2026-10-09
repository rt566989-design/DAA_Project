"""
Scenario and sample data generator for Employee Shift Scheduling.
Provides realistic hospital, corporate, and call-center staff configurations.
"""
import random
from typing import List, Dict, Tuple, Set
from .models import Employee, ShiftType, ConstraintRules


class ScenarioGenerator:
    """Generates realistic employee datasets and shift requirements."""

    NAMES = [
        "Aarav Sharma", "Priya Patel", "Rohan Verma", "Sneha Iyer",
        "Vikram Singh", "Ananya Gupta", "Rahul Nair", "Neha Joshi",
        "Aditya Kulkarni", "Pooja Mehta", "Karan Malhotra", "Riya Sen",
        "Arjun Das", "Tanvi Bhatia", "Devendra Rawat", "Meera Pillai",
        "Suresh Reddy", "Kavita Rao", "Deepak Chawla", "Shweta Kapoor",
        "Manish Tiwari", "Simran Kaur", "Nikhil Saxena", "Preeti Deshmukh"
    ]

    @classmethod
    def get_preset_scenario(cls, preset_name: str) -> Tuple[List[Employee], int, Dict[str, int], ConstraintRules]:
        """
        Returns (employees, days, shift_requirements, rules)
        Presets: 'small', 'medium', 'large'
        """
        preset_name = preset_name.lower()
        if preset_name == "small":
            # 6 employees, 7 days, requirements: M:1, E:1, N:1 (Total slots: 21 slots. 6 emps * 3.5 = 21)
            days = 7
            reqs = {ShiftType.MORNING.value: 1, ShiftType.EVENING.value: 1, ShiftType.NIGHT.value: 1}
            emps = cls.generate_employees(count=6, days=days, seed=42)
            rules = ConstraintRules(max_shifts_per_day=1, max_shifts_per_week=5, min_rest_night_to_morning=True)
            return emps, days, reqs, rules

        elif preset_name == "medium":
            # 12 employees, 7 days, requirements: M:2, E:2, N:1 (Total slots: 35 slots. 12 emps * ~3 = 35)
            days = 7
            reqs = {ShiftType.MORNING.value: 2, ShiftType.EVENING.value: 2, ShiftType.NIGHT.value: 1}
            emps = cls.generate_employees(count=12, days=days, seed=101)
            rules = ConstraintRules(max_shifts_per_day=1, max_shifts_per_week=5, min_rest_night_to_morning=True)
            return emps, days, reqs, rules

        elif preset_name == "large":
            # 20 employees, 7 days, requirements: M:4, E:3, N:2 (Total slots: 63 slots)
            days = 7
            reqs = {ShiftType.MORNING.value: 4, ShiftType.EVENING.value: 3, ShiftType.NIGHT.value: 2}
            emps = cls.generate_employees(count=20, days=days, seed=202)
            rules = ConstraintRules(max_shifts_per_day=1, max_shifts_per_week=5, min_rest_night_to_morning=True)
            return emps, days, reqs, rules

        else:
            return cls.get_preset_scenario("medium")

    @classmethod
    def generate_employees(
        cls,
        count: int,
        days: int = 7,
        max_shifts_per_week: int = 5,
        unavailable_prob: float = 0.15,
        seed: int = 42
    ) -> List[Employee]:
        rng = random.Random(seed)
        employees: List[Employee] = []

        shifts = ShiftType.all_shifts()
        name_pool = cls.NAMES.copy()

        for i in range(count):
            if i < len(name_pool):
                name = name_pool[i]
            else:
                name = f"Employee {i+1}"

            # Preferences: assign random ratings between 1 and 5
            prefs = {s: rng.randint(1, 5) for s in shifts}

            # Occasional unavailability (leaves)
            unav_days: Set[int] = set()
            for d in range(days):
                if rng.random() < unavailable_prob:
                    unav_days.add(d)

            emp = Employee(
                id=i + 1,
                name=name,
                max_shifts_per_week=max_shifts_per_week,
                preferences=prefs,
                unavailable_days=unav_days
            )
            employees.append(emp)

        return employees
