# 🏥 💼 Employee Shift Scheduling System (DAA Project)

> **Design and Analysis of Algorithms (DAA) Laboratory / Capstone Project**  
> A multi-paradigm comparative study solving the NP-Hard Employee Shift Scheduling Problem using **Greedy Heuristics**, **Backtracking (CSP with MRV)**, **Branch and Bound (DFBnB)**, and **Genetic Algorithms**.

---

## 🌟 Key Features

- **4 Core DAA Algorithmic Paradigms Implemented & Benchmarked:**
  1. ⚡ **Greedy Choice Heuristic**: Priority-based greedy choice with dynamic workload balancing.
  2. 🔍 **Backtracking (CSP with MRV & Forward Checking)**: Exhaustive search with fail-first variable ordering and domain pruning.
  3. 🌳 **Branch and Bound (DFBnB)**: Depth-first search with admissible lower-bounding function for globally optimal rosters.
  4. 🧬 **Genetic Algorithm**: Bio-inspired evolutionary metaheuristic with chromosome encoding, tournament selection, uniform crossover, and elitism.
- **Strict Real-World Constraints:**
  - Mandatory shift headcount coverage (Morning, Evening, Night).
  - Maximum 1 shift per employee per day (no double booking).
  - Ergonomic rest rule (Night shift $\to$ Morning shift transition strictly prohibited).
  - Weekly overtime cap and leave/unavailability request enforcement.
- **🔄 Automated Shift Change & Emergency Replacement Engine:**
  - **1-Click Auto Shift Change**: Automatically moves employee to target shift or identifies the optimal peer swap partner if full.
  - **Peer-to-Peer Shift Swapping**: Validates mutual constraint safety (rest periods, leaves, weekly limits) in $O(1)$ time and swaps shifts.
  - **Emergency Leave & Auto-Replacement**: When an employee reports sick or takes sudden leave, the system evaluates all candidates via multi-objective scoring (Fairness + Preferences) and assigns the best legal replacement.
  - **Constraint Guarantee**: Strictly prevents double-booking, ergonomic night $\to$ morning conflicts, and weekly hour violations.
- **Fairness & Multi-Objective Evaluation:**
  - **Jain's Fairness Index** ($J \in [1/N, 1.0]$) for measuring workload balance.
  - Employee shift preference satisfaction percentage.
  - Hard constraint violation detection and tracking.
- **Two User Interfaces:**
  - 🌐 **Interactive Streamlit Web Dashboard**: Live sliders, interactive roster grids, real-time benchmark charts, download to CSV / Excel, and dedicated **Auto Shift Change Workspace**.
  - 💻 **Rich Terminal CLI**: One-command side-by-side benchmark with terminal tables, Matplotlib graph exports, and automated shift-change simulation.
- **Academic Project Submission Material:**
  - Full formal project report in [DAA_PROJECT_REPORT.md](DAA_PROJECT_REPORT.md).
  - Top Viva Voce questions & answers prepared for lab evaluations.

---

## 📊 Algorithmic Complexity Comparison

| Algorithm | Paradigm | Time Complexity | Space Complexity | Optimality Guarantee |
| :--- | :--- | :--- | :--- | :--- |
| **Greedy Choice Heuristic** | Greedy / Priority Search | $O(D \cdot S \cdot N \log N)$ | $O(D \cdot S + N)$ | Local Optima |
| **Backtracking (CSP)** | Exhaustive Search + Pruning | Worst $O(N^{D \cdot S})$, Pruned $O(b^d)$ | $O(D \cdot S)$ | Feasibility Guaranteed |
| **Branch and Bound** | Optimal State-Space Tree | $O(b^d)$ with Admissible Bound | $O(D \cdot S)$ | Global Optimum |
| **Genetic Algorithm** | Evolutionary Metaheuristic | $O(G \cdot P \cdot D \cdot S)$ | $O(P \cdot D \cdot S)$ | Near-Optimal |

---

## 🚀 Quick Start Guide

### 1. Requirements & Installation
The project runs on Python 3.10+ (tested on Python 3.14). All required libraries are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Web Dashboard
Launch the web interface in your browser:

```bash
streamlit run app.py
```

*The web application allows you to adjust parameters (number of staff, days, shift coverage requirements), generate rosters, visualize heatmaps, compare all 4 algorithms head-to-head, and export schedules.*

### 3. Run the Command-Line Interface (CLI) Demo
Run a side-by-side benchmark comparison and generate PNG visualizations:

```bash
python main.py --demo
```

Outputs will be saved automatically to the `output/` directory:
- `output/benchmark_comparison.png` (4-panel comparison graph)
- `output/schedule_heatmap.png` (Employee shift roster heatmap)
- `output/schedule_roster.csv` (CSV schedule export)
- `output/schedule_report.xlsx` (Multi-sheet Excel report)

### 4. Run Automated Shift Change & Replacement Demo
To run the automated shift change, peer swap, and emergency leave simulation:

```bash
python main.py --shift-change
```

### 5. Run Scaling Benchmark
To evaluate how algorithms scale across Small, Medium, and Large scenarios:

```bash
python main.py --benchmark
```

### 6. Run Unit Tests
To verify constraint enforcement, algorithmic correctness, and shift management:

```bash
python -m unittest discover tests
```

---

## 📁 Project Directory Structure

```
DAA_Experiment1_code/
├── app.py                      # Interactive Streamlit Web Application (5 Tabs + Auto Shift Change Workspace)
├── main.py                     # Command-line driver, benchmark runner & shift-change demo
├── requirements.txt            # Python dependencies
├── DAA_PROJECT_REPORT.md       # Comprehensive academic lab & capstone project report
├── README.md                   # Project documentation & user guide
│
├── core/
│   ├── __init__.py
│   ├── models.py               # Employee, Shift, Schedule, and Constraint definitions
│   ├── evaluator.py            # Hard constraint validator & Jain's Fairness Index calculator
│   ├── shift_manager.py        # Automated shift change, peer swaps & emergency replacement engine
│   └── data_generator.py       # Realistic hospital & tech support dataset generator
│
├── algorithms/
│   ├── __init__.py
│   ├── greedy.py               # Greedy Choice Heuristic with Workload Balancing
│   ├── backtracking.py         # Backtracking CSP with MRV and Forward Checking
│   ├── branch_and_bound.py     # Depth-First Branch & Bound with Lower Bounding
│   └── genetic.py              # Genetic Algorithm with Tournament Selection & Elitism
│
├── utils/
│   ├── __init__.py
│   ├── visualizer.py           # Publication-grade Matplotlib charts & heatmaps
│   └── exporter.py             # CSV, Excel, and JSON exporters
│
├── tests/
│   ├── __init__.py
│   ├── test_algorithms.py      # Unit tests verifying all 4 offline schedulers
│   └── test_shift_manager.py   # Unit tests verifying auto-shift change, swaps, and emergency leaves
│
└── output/                     # Auto-generated benchmark plots, heatmaps, and spreadsheets
```

---

## 🎓 Viva Voce & Presentation Cheat-Sheet

1. **Why is Shift Scheduling NP-Hard?**
   - It reduces from the **Exact Cover by 3-Sets (X3C)** problem. For $N$ workers and $K$ slots, the combinatorial space is $N^K$.
2. **What is MRV (Minimum Remaining Values)?**
   - It is a CSP variable selection heuristic ("fail-first" principle). By assigning the shift slot with the fewest legal workers first, failing branches are pruned at the top of the search tree.
3. **What is Jain's Fairness Index?**
   - A scale-independent metric $J = \frac{(\sum w_i)^2}{N \sum w_i^2} \in [1/N, 1.0]$. A score of $1.0$ indicates equal workload balance across all employees.
4. **How does Branch & Bound prune?**
   - It calculates an admissible lower bound $LB(u)$ for partial schedule $u$. If $LB(u) \ge \text{BestCostKnown}$, subtree $u$ is pruned because it can never yield an optimal solution.
5. **How does Dynamic Shift Change work without re-running solvers?**
   - Performs localized $O(1)$ constraint checking across all hard rules and finds the optimal replacement candidate in $O(N)$ time via multi-objective greedy choice scoring, preserving 100% feasibility and fairness balance in sub-milliseconds.

---
*Developed for Design and Analysis of Algorithms (DAA) Laboratory & Academic Submission.*

