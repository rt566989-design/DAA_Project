"""
Export utilities for saving schedules to CSV, Excel (.xlsx), and JSON.
"""
import json
from typing import List, Dict, Optional
import pandas as pd
from core.models import Schedule, Employee
from core.evaluator import EvaluationResult


class ScheduleExporter:
    """Exports generated schedules and evaluation summaries to file formats."""

    @staticmethod
    def export_to_csv(
        schedule: Schedule,
        employees: List[Employee],
        filepath: str
    ) -> None:
        """Exports employee-centric roster to CSV."""
        df = schedule.to_employee_roster(employees)
        df.to_csv(filepath, index=False)

    @staticmethod
    def export_to_excel(
        schedule: Schedule,
        employees: List[Employee],
        evaluation: Optional[EvaluationResult],
        filepath: str
    ) -> None:
        """Exports multiple sheets to an Excel workbook: Roster, Matrix, and Evaluation Report."""
        emp_map = {e.id: e for e in employees}
        roster_df = schedule.to_employee_roster(employees)
        matrix_df = schedule.to_dataframe(emp_map)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            roster_df.to_excel(writer, sheet_name="Employee Roster", index=False)
            matrix_df.to_excel(writer, sheet_name="Shift Matrix", index=True)

            if evaluation is not None:
                summary_data = {
                    "Metric": [
                        "Is Feasible / Valid",
                        "Hard Violations Count",
                        "Coverage %",
                        "Total Shifts Assigned",
                        "Required Shifts Total",
                        "Preference Satisfaction %",
                        "Workload Mean",
                        "Workload Std Dev",
                        "Jain's Fairness Index",
                        "Composite Fitness Score"
                    ],
                    "Value": [
                        "Yes" if evaluation.is_valid else "No",
                        evaluation.hard_violations_count,
                        f"{evaluation.coverage_percentage}%",
                        evaluation.total_shifts_assigned,
                        evaluation.required_shifts_total,
                        f"{evaluation.preference_satisfaction_pct}%",
                        evaluation.workload_mean,
                        evaluation.workload_std_dev,
                        evaluation.jains_fairness_index,
                        evaluation.composite_fitness_score
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Evaluation Report", index=False)

    @staticmethod
    def export_to_json(
        schedule: Schedule,
        employees: List[Employee],
        filepath: str
    ) -> None:
        """Exports schedule assignments to structured JSON."""
        emp_map = {e.id: e for e in employees}
        data = {
            "days": schedule.days,
            "shifts": schedule.shifts,
            "assignments": []
        }
        for d in range(schedule.days):
            for s in schedule.shifts:
                eids = schedule.get_assigned(d, s)
                data["assignments"].append({
                    "day": d + 1,
                    "shift": s,
                    "employee_ids": eids,
                    "employees": [emp_map[eid].name for eid in eids if eid in emp_map]
                })

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
