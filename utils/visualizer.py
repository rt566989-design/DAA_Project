"""
Visualization utilities for Employee Shift Scheduling.
Provides publication-grade charts for schedules, benchmark comparisons, and asymptotic scaling.
"""
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from core.models import Schedule, Employee, ShiftType
from core.evaluator import EvaluationResult


class ScheduleVisualizer:
    """Creates clear, color-coded Matplotlib charts for shift schedules and algorithmic benchmarks."""

    # Color palette
    SHIFT_COLORS = {
        "Morning": "#38bdf8",   # Soft Sky Blue
        "Evening": "#fbbf24",   # Amber / Orange
        "Night": "#818cf8",     # Indigo / Violet
        "OFF": "#e2e8f0"        # Neutral Slate Gray
    }

    @classmethod
    def plot_schedule_heatmap(
        cls,
        schedule: Schedule,
        employees: List[Employee],
        title: str = "Employee Weekly Shift Roster",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualizes an Employee x Day shift roster grid."""
        n_emps = len(employees)
        days = schedule.days
        day_labels = [f"Day {d+1}" for d in range(days)]
        emp_names = [e.name for e in employees]

        # Shift to numeric code for plotting: OFF=0, Morning=1, Evening=2, Night=3
        code_map = {"OFF": 0, "Morning": 1, "Evening": 2, "Night": 3}
        matrix = np.zeros((n_emps, days), dtype=int)

        for i, emp in enumerate(employees):
            for d in range(days):
                shift = schedule.get_employee_shift_for_day(emp.id, d)
                matrix[i, d] = code_map.get(shift, 0)

        fig, ax = plt.subplots(figsize=(max(8, days * 1.2), max(5, n_emps * 0.45)), dpi=120)

        # Plot cells
        for i in range(n_emps):
            for d in range(days):
                code = matrix[i, d]
                shift_name = "OFF" if code == 0 else list(code_map.keys())[code]
                bg_color = cls.SHIFT_COLORS.get(shift_name, "#ffffff")
                rect = plt.Rectangle((d - 0.45, i - 0.45), 0.9, 0.9, color=bg_color, ec="#94a3b8", lw=1.2)
                ax.add_patch(rect)
                text_color = "#0f172a"
                label = shift_name[:1] if shift_name != "OFF" else "-"
                ax.text(d, i, label, ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

        ax.set_xlim(-0.6, days - 0.4)
        ax.set_ylim(-0.6, n_emps - 0.4)
        ax.set_xticks(range(days))
        ax.set_xticklabels(day_labels, fontweight="bold", fontsize=10)
        ax.set_yticks(range(n_emps))
        ax.set_yticklabels(emp_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

        # Legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=cls.SHIFT_COLORS[s], label=s)
            for s in ["Morning", "Evening", "Night", "OFF"]
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=True)
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig

    @classmethod
    def plot_benchmark_comparison(
        cls,
        benchmark_results: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plots a 4-panel comparison matrix:
        1. Execution Time (ms)
        2. Preference Satisfaction (%)
        3. Jain's Fairness Index
        4. Hard Constraint Violations
        """
        algos = [res["algorithm"] for res in benchmark_results]
        times_ms = [res["execution_time_sec"] * 1000 for res in benchmark_results]
        satisfaction = [res["evaluation"].preference_satisfaction_pct for res in benchmark_results]
        fairness = [res["evaluation"].jains_fairness_index for res in benchmark_results]
        violations = [res["evaluation"].hard_violations_count for res in benchmark_results]

        palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=120)
        fig.suptitle("DAA Algorithmic Benchmark Comparison", fontsize=15, fontweight="bold")

        # 1. Execution Time
        bars = axes[0, 0].bar(algos, times_ms, color=palette[:len(algos)], edgecolor="#334155")
        axes[0, 0].set_title("Runtime / Execution Time (ms) [Lower is Better]", fontweight="bold", fontsize=10)
        axes[0, 0].set_ylabel("Milliseconds (ms)")
        axes[0, 0].tick_params(axis='x', rotation=18)
        for bar in bars:
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.1f}ms", ha='center', va='bottom', fontsize=8)

        # 2. Preference Satisfaction
        bars = axes[0, 1].bar(algos, satisfaction, color=palette[:len(algos)], edgecolor="#334155")
        axes[0, 1].set_title("Preference Satisfaction (%) [Higher is Better]", fontweight="bold", fontsize=10)
        axes[0, 1].set_ylabel("Satisfaction %")
        axes[0, 1].set_ylim(0, 105)
        axes[0, 1].tick_params(axis='x', rotation=18)
        for bar in bars:
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8)

        # 3. Jain's Fairness Index
        bars = axes[1, 0].bar(algos, fairness, color=palette[:len(algos)], edgecolor="#334155")
        axes[1, 0].set_title("Jain's Fairness Index [1.0 = Perfect Balance]", fontweight="bold", fontsize=10)
        axes[1, 0].set_ylabel("Fairness (0 to 1.0)")
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].tick_params(axis='x', rotation=18)
        for bar in bars:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)

        # 4. Hard Violations
        bars = axes[1, 1].bar(algos, violations, color=["#ef4444" if v > 0 else "#22c55e" for v in violations], edgecolor="#334155")
        axes[1, 1].set_title("Hard Constraint Violations [0 = Feasible]", fontweight="bold", fontsize=10)
        axes[1, 1].set_ylabel("Count of Violations")
        axes[1, 1].tick_params(axis='x', rotation=18)
        for bar in bars:
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(bar.get_height())}", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig

    @classmethod
    def plot_workload_distribution(
        cls,
        eval_result: EvaluationResult,
        title: str = "Employee Workload Distribution (Total Shifts Assigned)",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """Plots the number of shifts assigned per employee."""
        emp_names = list(eval_result.workload_distribution.keys())
        counts = list(eval_result.workload_distribution.values())

        fig, ax = plt.subplots(figsize=(max(8, len(emp_names) * 0.45), 4.5), dpi=120)
        bars = ax.bar(emp_names, counts, color="#0284c7", edgecolor="#0369a1")
        ax.axhline(eval_result.workload_mean, color="#ef4444", linestyle="--", label=f"Mean: {eval_result.workload_mean:.1f} shifts")
        ax.set_ylabel("Total Shifts Worked", fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(loc="upper right")

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig
