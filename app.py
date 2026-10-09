"""
Employee Shift Scheduling System - Modern Streamlit Interactive Web Application
Design and Analysis of Algorithms (DAA) Laboratory / Capstone Project.
"""
import io
import time
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.models import ShiftType, Employee, ConstraintRules, Schedule
from core.evaluator import ScheduleEvaluator, EvaluationResult
from core.data_generator import ScenarioGenerator
from algorithms.greedy import GreedyScheduler
from algorithms.backtracking import BacktrackingScheduler
from algorithms.branch_and_bound import BranchAndBoundScheduler
from algorithms.genetic import GeneticAlgorithmScheduler
from utils.visualizer import ScheduleVisualizer
from utils.exporter import ScheduleExporter

# Set page config
st.set_page_config(
    page_title="Employee Shift Scheduling - DAA Project",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-blue { background: #dbeafe; color: #1e40af; }
    .badge-emerald { background: #d1fae5; color: #065f46; }
    .badge-amber { background: #fef3c7; color: #92400e; }
    .badge-purple { background: #ede9fe; color: #5b21b6; }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.title("⚙️ Scenario & Parameters")

scenario_choice = st.sidebar.selectbox(
    "Choose Preset Scenario",
    ["Medium (12 Staff, 7 Days)", "Small (6 Staff, 7 Days)", "Large (20 Staff, 7 Days)", "Custom Configuration"],
    index=0
)

if "Custom" in scenario_choice:
    n_employees = st.sidebar.slider("Number of Employees", 4, 30, 10)
    n_days = st.sidebar.slider("Schedule Duration (Days)", 3, 14, 7)
    req_m = st.sidebar.number_input("Morning Shift Required Staff", min_value=1, max_value=10, value=2)
    req_e = st.sidebar.number_input("Evening Shift Required Staff", min_value=1, max_value=10, value=2)
    req_n = st.sidebar.number_input("Night Shift Required Staff", min_value=1, max_value=10, value=1)
    max_weekly = st.sidebar.slider("Max Shifts per Employee", 1, 7, 5)

    employees = ScenarioGenerator.generate_employees(count=n_employees, days=n_days, max_shifts_per_week=max_weekly)
    days = n_days
    reqs = {ShiftType.MORNING.value: req_m, ShiftType.EVENING.value: req_e, ShiftType.NIGHT.value: req_n}
    rules = ConstraintRules(max_shifts_per_day=1, max_shifts_per_week=max_weekly, min_rest_night_to_morning=True)
else:
    preset_key = "small" if "Small" in scenario_choice else ("large" if "Large" in scenario_choice else "medium")
    employees, days, reqs, rules = ScenarioGenerator.get_preset_scenario(preset_key)

st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Constraints Enforced")
rules.min_rest_night_to_morning = st.sidebar.checkbox("Ergonomic Rest (No Night -> Morning next day)", value=rules.min_rest_night_to_morning)
rules.respect_unavailability = st.sidebar.checkbox("Respect Leave / Unavailability Requests", value=rules.respect_unavailability)
rules.max_shifts_per_day = st.sidebar.number_input("Max Shifts per Employee per Day", min_value=1, max_value=2, value=1)

# Main Title & Badges
st.markdown('<div class="main-title">📅 Employee Shift Scheduling System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Design and Analysis of Algorithms (DAA) Capstone Project • NP-Hard Optimization & Multi-Paradigm Comparative Study</div>', unsafe_allow_html=True)

st.markdown("""
<div>
    <span class="badge-pill badge-blue">⚡ Greedy Choice Heuristic</span>
    <span class="badge-pill badge-emerald">🔍 Backtracking CSP (MRV + Pruning)</span>
    <span class="badge-pill badge-purple">🌳 Branch and Bound (DFBnB)</span>
    <span class="badge-pill badge-amber">🧬 Genetic Algorithm (Metaheuristic)</span>
</div>
<br>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Generate & View Schedule",
    "⚔️ Head-to-Head Algorithm Benchmark",
    "🔬 DAA Algorithmic Deep Dive",
    "🎓 Viva Voce & Project Report"
])


# ==============================================================================
# TAB 1: SCHEDULE GENERATOR
# ==============================================================================
with tab1:
    col_ctrl, col_btn = st.columns([3, 1])
    with col_ctrl:
        selected_algo = st.selectbox(
            "Select DAA Algorithm to Execute",
            [
                "Greedy Heuristic [O(D * S * N log N)]",
                "Backtracking CSP with MRV [O(N^(D*S)) -> Pruned]",
                "Branch and Bound DFBnB [O(b^d) with Admissible Bound]",
                "Genetic Algorithm [O(G * P * D * S)]"
            ]
        )
    with col_btn:
        st.write("")
        st.write("")
        run_button = st.button("🚀 Generate Schedule", type="primary", use_container_width=True)

    if "schedule_res" not in st.session_state or run_button:
        # Instantiate algorithm
        if "Greedy" in selected_algo:
            solver = GreedyScheduler()
        elif "Backtracking" in selected_algo:
            solver = BacktrackingScheduler(time_limit_sec=4.0)
        elif "Branch and Bound" in selected_algo:
            solver = BranchAndBoundScheduler(time_limit_sec=3.0)
        else:
            solver = GeneticAlgorithmScheduler(population_size=40, generations=30, seed=42)

        with st.spinner(f"Executing {solver.name}..."):
            sched, meta = solver.solve(employees, days, reqs, rules)
            eval_res = ScheduleEvaluator.evaluate(sched, employees, reqs, rules)
            st.session_state["schedule_res"] = (sched, meta, eval_res, solver.name)

    sched, meta, eval_res, algo_name = st.session_state["schedule_res"]

    # Top KPI Metrics Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            label="Feasibility Status",
            value="Valid (Feasible)" if eval_res.is_valid else f"{eval_res.hard_violations_count} Violations",
            delta="Zero Violations" if eval_res.is_valid else "Violations detected",
            delta_color="normal" if eval_res.is_valid else "inverse"
        )
    with m2:
        st.metric(
            label="Execution Runtime",
            value=f"{meta['execution_time_sec'] * 1000:.2f} ms",
            help="Total CPU time taken by algorithm solver"
        )
    with m3:
        st.metric(
            label="Preference Satisfaction",
            value=f"{eval_res.preference_satisfaction_pct:.1f}%",
            delta=f"{eval_res.total_preference_score}/{eval_res.max_possible_preference} pts"
        )
    with m4:
        st.metric(
            label="Jain's Fairness Index",
            value=f"{eval_res.jains_fairness_index:.4f}",
            help="1.000 = Perfectly fair, balanced shift distribution"
        )
    with m5:
        st.metric(
            label="Coverage Ratio",
            value=f"{eval_res.coverage_percentage:.1f}%",
            delta=f"{eval_res.total_shifts_assigned}/{eval_res.required_shifts_total} slots"
        )

    if not eval_res.is_valid:
        with st.expander("⚠️ View Constraint Violations Details", expanded=True):
            for v in eval_res.hard_violations:
                st.warning(v)

    st.markdown("---")
    st.subheader(f"📅 Schedule Roster ({algo_name})")

    view_type = st.radio("Display View", ["Employee Roster (Table)", "Visual Heatmap Chart", "Shift Coverage Matrix"], horizontal=True)

    emp_map = {e.id: e for e in employees}
    roster_df = sched.to_employee_roster(employees)

    if view_type == "Employee Roster (Table)":
        st.dataframe(roster_df, use_container_width=True)
    elif view_type == "Visual Heatmap Chart":
        fig_heat = ScheduleVisualizer.plot_schedule_heatmap(sched, employees, title=f"Schedule Grid - {algo_name}")
        st.pyplot(fig_heat)
    else:
        matrix_df = sched.to_dataframe(emp_map)
        st.dataframe(matrix_df, use_container_width=True)

    # Workload histogram & preference breakdown
    col_dist, col_pref = st.columns(2)
    with col_dist:
        st.subheader("📊 Workload Distribution")
        fig_dist = ScheduleVisualizer.plot_workload_distribution(eval_res)
        st.pyplot(fig_dist)

    with col_pref:
        st.subheader("⭐ Employee Preference Matrix")
        pref_data = []
        for emp in employees:
            pref_data.append({
                "Employee": emp.name,
                "Morning": f"{emp.get_preference('Morning')} ★",
                "Evening": f"{emp.get_preference('Evening')} ★",
                "Night": f"{emp.get_preference('Night')} ★",
                "Leaves": f"{len(emp.unavailable_days)} days"
            })
        st.dataframe(pd.DataFrame(pref_data), use_container_width=True, height=260)

    # Export options
    st.markdown("---")
    st.subheader("💾 Export & Download Schedule")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        csv_buffer = roster_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Roster as CSV",
            data=csv_buffer,
            file_name=f"shift_roster_{algo_name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d_col2:
        excel_buffer = io.BytesIO()
        ScheduleExporter.export_to_excel(sched, employees, eval_res, excel_buffer)
        st.download_button(
            label="📊 Download Full Excel Report (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"schedule_report_{algo_name.replace(' ', '_').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


# ==============================================================================
# TAB 2: HEAD-TO-HEAD BENCHMARK
# ==============================================================================
with tab2:
    st.header("⚔️ DAA Head-to-Head Algorithmic Benchmark")
    st.markdown("""
    Compare the 4 algorithmic paradigms across **Execution Runtime**, **Solution Quality (Preferences)**, 
    **Jain's Fairness Index**, and **Hard Constraint Satisfaction**.
    """)

    if st.button("⚡ Run Full Benchmark Comparison", type="primary"):
        schedulers = [
            GreedyScheduler(),
            BacktrackingScheduler(time_limit_sec=3.5),
            BranchAndBoundScheduler(time_limit_sec=3.0),
            GeneticAlgorithmScheduler(population_size=40, generations=30, seed=42)
        ]

        bench_results = []
        progress_bar = st.progress(0)
        for idx, sch in enumerate(schedulers):
            s_sched, s_meta = sch.solve(employees, days, reqs, rules)
            s_eval = ScheduleEvaluator.evaluate(s_sched, employees, reqs, rules)
            bench_results.append({
                "algorithm": sch.name,
                "execution_time_sec": s_meta["execution_time_sec"],
                "evaluation": s_eval,
                "metadata": s_meta
            })
            progress_bar.progress((idx + 1) / len(schedulers))

        st.session_state["benchmark_data"] = bench_results

    if "benchmark_data" in st.session_state:
        b_data = st.session_state["benchmark_data"]

        # Summary Table
        summary_rows = []
        for r in b_data:
            summary_rows.append({
                "Algorithm": r["algorithm"],
                "Runtime (ms)": f"{r['execution_time_sec'] * 1000:.2f} ms",
                "Hard Violations": r["evaluation"].hard_violations_count,
                "Preference %": f"{r['evaluation'].preference_satisfaction_pct:.1f}%",
                "Jain's Fairness Index": f"{r['evaluation'].jains_fairness_index:.4f}",
                "Composite Fitness": r["evaluation"].composite_fitness_score,
                "Time Complexity": r["metadata"].get("time_complexity", "N/A"),
                "Space Complexity": r["metadata"].get("space_complexity", "N/A")
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # Plot 4-panel comparison
        fig_bench = ScheduleVisualizer.plot_benchmark_comparison(b_data)
        st.pyplot(fig_bench)

        st.markdown("### 🏆 Algorithmic Trade-off Analysis")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.info("""
            **⚡ Greedy Choice**
            - **Speed:** Ultra Fast (<1 ms)
            - **Optimality:** Local Optima
            - **Best for:** Real-time online dispatch, large scales ($N > 100$).
            """)
        with c2:
            st.success("""
            **🔍 Backtracking (CSP)**
            - **Speed:** Fast for small instances
            - **Optimality:** 100% Feasible guarantee
            - **Best for:** Strict regulatory compliance, small-to-medium teams.
            """)
        with c3:
            st.warning("""
            **🌳 Branch & Bound**
            - **Speed:** Exponential, pruned
            - **Optimality:** Globally Optimal
            - **Best for:** Offline weekly rosters where absolute perfection is needed.
            """)
        with c4:
            st.error(r"""
            **🧬 Genetic Algorithm**
            - **Speed:** Predictable $O(G \cdot P)$
            - **Optimality:** High-quality metaheuristic
            - **Best for:** Large, highly overconstrained enterprise rosters.
            """)


# ==============================================================================
# TAB 3: DAA ALGORITHMIC DEEP DIVE
# ==============================================================================
with tab3:
    st.header("🔬 DAA Algorithmic Deep Dive & Formal Complexity")

    st.markdown(r"""
    ### 1. Mathematical Problem Formulation
    The **Employee Shift Scheduling Problem (ESSP)** is a well-known combinatorial optimization problem closely related to the *Nurse Scheduling Problem (NSP)*.

    Let:
    - $E = \{e_1, e_2, \dots, e_N\}$ be the set of $N$ employees.
    - $D = \{1, 2, \dots, T\}$ be the set of $T$ scheduling days.
    - $S = \{\text{Morning}, \text{Evening}, \text{Night}\}$ be the shift types.
    - $R_{s}$ be the required headcount for shift $s \in S$.
    - $P_{e, s} \in [1, 5]$ be the preference score of employee $e$ for shift $s$.

    **Decision Variable:**
    $$x_{e, d, s} \in \{0, 1\}$$
    where $x_{e, d, s} = 1$ if employee $e$ is assigned to shift $s$ on day $d$, else $0$.

    ---
    ### 2. Constraints Formulation
    #### A. Hard Constraints (Feasibility Criteria)
    1. **Shift Demand / Coverage Constraint:**
       $$\\forall d \in D, \\forall s \in S: \\sum_{e \\in E} x_{e, d, s} = R_{s}$$
    2. **Daily Limit (Max 1 shift per day):**
       $$\\forall e \\in E, \\forall d \\in D: \\sum_{s \\in S} x_{e, d, s} \\le 1$$
    3. **Ergonomic Rest Period (Forbidden Night $\\to$ Morning Transition):**
       $$\\forall e \\in E, \\forall d \\in \\{1, \\dots, T-1\\}: x_{e, d, \\text{Night}} + x_{e, d+1, \\text{Morning}} \\le 1$$
    4. **Weekly Overtime Limit:**
       $$\\forall e \\in E: \\sum_{d \\in D} \\sum_{s \\in S} x_{e, d, s} \\le \\text{MaxShifts}(e)$$
    5. **Unavailability / Leave Requests:**
       $$\\forall e \\in E, \\forall d \\in \\text{Leaves}(e), \\forall s \\in S: x_{e, d, s} = 0$$

    #### B. Soft Constraints (Multi-Objective Optimization)
    1. **Maximize Preference Satisfaction:**
       $$\\max Z_{pref} = \\sum_{e \\in E} \\sum_{d \\in D} \\sum_{s \\in S} P_{e, s} \\cdot x_{e, d, s}$$
    2. **Workload Fairness (Jain's Fairness Index):**
       $$J(X) = \\frac{\\left(\\sum_{i=1}^N x_i\\right)^2}{N \\sum_{i=1}^N x_i^2}$$
       where $x_i$ is the total shifts assigned to employee $i$. $J(X) = 1.0$ indicates perfect balance.

    ---
    ### 3. NP-Hardness Classification
    > **Theorem:** The Employee Shift Scheduling Problem with hard constraints is **NP-Complete in decision form** and **NP-Hard in optimization form**.
    
    *Proof Sketch:* Reduction from the **Exact Cover by 3-Sets (X3C)** problem or the **Timetabling Problem**. Setting shift requirements as sets of triples and restricting employees maps directly to exact covering of universe elements without overlap. Hence, no polynomial-time exact algorithm exists unless $P = NP$.

    ---
    ### 4. Asymptotic Complexity Comparison Matrix
    """)

    complexity_df = pd.DataFrame({
        "Algorithm": ["Greedy Choice", "Backtracking (CSP)", "Branch & Bound (DFBnB)", "Genetic Algorithm"],
        "Paradigm": ["Heuristic / Greedy Choice", "Exhaustive State-Space Tree", "Optimal Tree Search with Bounding", "Evolutionary Metaheuristic"],
        "Worst-Case Time": ["O(D * S * N log N)", "O(N^(D * S))", "O(b^d) where b=N, d=D*S", "O(G * P * (D * S))"],
        "Average Pruned Time": ["O(D * S * N log N)", "O(b^d) with b << N via MRV", "O(b^d) with tight pruning", "O(G * P * (D * S))"],
        "Space Complexity": ["O(D * S + N)", "O(D * S) recursion stack", "O(D * S)", "O(P * D * S) population"],
        "Optimality Guarantee": ["No (Local Optima)", "Feasibility Guaranteed", "Global Optimum Guaranteed", "Near-Optimal / Approximate"]
    })
    st.table(complexity_df)


# ==============================================================================
# TAB 4: VIVA VOCE & ACADEMIC REPORT
# ==============================================================================
with tab4:
    st.header("🎓 Academic Project Report & DAA Viva Voce Questions")

    st.markdown("""
    Use these questions and answers to prepare for internal reviews, external lab examinations, and viva voce.
    """)

    viva_qa = [
        ("Q1: What makes Employee Shift Scheduling an algorithmic challenge?",
         "It is an NP-Hard combinatorial optimization problem. For a standard team of 15 employees over 7 days with 3 shifts, the theoretical search space is 15^(21) ≈ 5.8 x 10^24 combinations, making brute-force enumeration impossible."),

        ("Q2: How does the MRV (Minimum Remaining Values) heuristic improve Backtracking?",
         "MRV operates on the 'fail-first' principle. By picking the shift slot that has the fewest eligible employees left in its domain, the search tree fails early at the top levels rather than wasting millions of recursive calls exploring doomed subtrees."),

        ("Q3: What is Forward Checking and how does it prevent dead ends?",
         "After assigning an employee to a slot, Forward Checking inspects all future unassigned slots. If assigning employee E causes any future slot's legal candidate count to drop to 0, the algorithm backtracks immediately without making recursive calls."),

        ("Q4: How does Branch and Bound guarantee optimality compared to Greedy?",
         "Greedy makes irreversible local decisions that can lead to suboptimal schedules or dead ends. Branch and Bound maintains an admissible lower bound on the objective cost. It explores the search space systematically and prunes entire subtrees whenever the lower bound exceeds the best-known solution, guaranteeing that the final solution is globally optimal."),

        ("Q5: What is Jain's Fairness Index and why is it preferred over standard deviation?",
         "Jain's Fairness Index, J = (sum(x_i))^2 / (n * sum(x_i^2)), is a dimensionless, scale-independent metric bounded strictly between 1/n and 1.0. A score of 1.0 means every employee has an identical workload. Unlike variance, it is independent of population size and scale."),

        ("Q6: Explain the Genetic Algorithm chromosome encoding and fitness function.",
         "Each chromosome is a list of length equal to total required shift slots (D * S * req), where each gene stores an employee ID. The fitness function rewards preference scores and Jain's fairness while applying heavy penalties for hard constraint violations (double-booking, night-to-morning transitions)."),

        ("Q7: When should an enterprise choose Greedy over Branch & Bound in production?",
         "When speed and scalability dominate. In large-scale hospital networks or dispatch centers with hundreds of employees and real-time shifts, Branch and Bound hits combinatorial explosion. Greedy executes in sub-milliseconds with acceptable heuristics."),

        ("Q8: What is the space complexity advantage of Depth-First Branch and Bound (DFBnB)?",
         "Best-First Branch and Bound stores all active frontier nodes in a priority queue, requiring exponential O(b^d) memory. Depth-First Branch and Bound (DFBnB) only stores the current branch on the stack, keeping space bounded to linear O(D * S).")
    ]

    for question, answer in viva_qa:
        with st.expander(f"📌 {question}"):
            st.write(answer)
