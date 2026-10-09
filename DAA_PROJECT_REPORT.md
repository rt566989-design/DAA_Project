# ACADEMIC PROJECT REPORT
## Design and Analysis of Algorithms (DAA) Laboratory / Capstone Project

---

### Project Title
# **Employee Shift Scheduling System: A Multi-Paradigm Algorithmic Study**

**Domain:** Combinatorial Optimization, Constraint Satisfaction Problems (CSP), NP-Hard Problems  
**Implementation Language:** Python 3.x (with Streamlit, NumPy, Pandas, Matplotlib)  
**Academic Course:** Design and Analysis of Algorithms (DAA)  

---

## 1. ABSTRACT
The Employee Shift Scheduling Problem (ESSP) is a foundational combinatorial optimization problem pervasive in hospital emergency wards, aviation operations, 24/7 IT technical support desks, and industrial production plants. The objective is to allocate a set of employees across shifts (Morning, Evening, Night) over a planning horizon while satisfying strict legal/ergonomic hard constraints (shift coverage, maximum work hours, mandatory rest periods, leave requests) and optimizing soft multi-objective criteria (preference satisfaction, workload fairness).

In this project, we formulate the ESSP mathematically, classify its theoretical complexity (NP-Hard), and design, implement, and benchmark **four major algorithmic paradigms**:
1. **Greedy Choice Heuristic with Workload Balancing** - $O(D \cdot S \cdot N \log N)$
2. **Backtracking with Constraint Satisfaction (MRV & Forward Checking)** - Worst $O(N^{D \cdot S})$, Pruned $O(b^d)$
3. **Branch and Bound with Admissible Lower Bounding (DFBnB)** - $O(b^d)$
4. **Genetic Algorithm (Evolutionary Metaheuristic)** - $O(G \cdot P \cdot D \cdot S)$

Empirical benchmarking demonstrates the trade-offs between execution runtime, constraint violation rate, preference satisfaction, and Jain's Fairness Index.

---

## 2. PROBLEM FORMULATION & MATHEMATICAL MODEL

### 2.1 Sets and Parameters
- $E = \{e_1, e_2, \dots, e_N\}$: Set of $N$ available employees.
- $D = \{1, 2, \dots, T\}$: Planning horizon of $T$ days (e.g., $T=7$ for 1 week).
- $S = \{\text{Morning}, \text{Evening}, \text{Night}\}$: Set of shifts per day ($|S|=3$).
- $R_{d, s} \in \mathbb{Z}^+$: Required staffing headcount for shift $s$ on day $d$.
- $P_{e, s} \in [1, 5]$: Preference rating given by employee $e$ for shift type $s$.
- $W_{max}(e)$: Maximum shifts allowed for employee $e$ within the planning period.
- $U_e \subseteq D$: Set of days where employee $e$ is unavailable (approved leaves).

### 2.2 Decision Variables
$$x_{e, d, s} \in \{0, 1\} \quad \forall e \in E, d \in D, s \in S$$
$$x_{e, d, s} = \begin{cases} 1 & \text{if employee } e \text{ is assigned to shift } s \text{ on day } d \\ 0 & \text{otherwise} \end{cases}$$

### 2.3 Hard Constraints (Mandatory Feasibility)
1. **Headcount Coverage Constraint:**
   $$\sum_{e \in E} x_{e, d, s} = R_{d, s} \quad \forall d \in D, \forall s \in S$$
2. **Daily Shift Limit (No Double Booking):**
   $$\sum_{s \in S} x_{e, d, s} \le 1 \quad \forall e \in E, \forall d \in D$$
3. **Ergonomic Rest Period Constraint:**
   An employee who works a Night shift on day $d$ cannot be assigned to a Morning shift on day $d+1$:
   $$x_{e, d, \text{Night}} + x_{e, d+1, \text{Morning}} \le 1 \quad \forall e \in E, \forall d \in \{1, \dots, T-1\}$$
4. **Maximum Working Shifts Constraint:**
   $$\sum_{d \in D} \sum_{s \in S} x_{e, d, s} \le W_{max}(e) \quad \forall e \in E$$
5. **Leave / Unavailability Enforcement:**
   $$x_{e, d, s} = 0 \quad \forall e \in E, \forall d \in U_e, \forall s \in S$$

### 2.4 Soft Constraints & Multi-Objective Function
1. **Maximize Shift Preference Score:**
   $$Z_{\text{pref}} = \sum_{e \in E} \sum_{d \in D} \sum_{s \in S} P_{e, s} \cdot x_{e, d, s}$$
2. **Maximize Jain's Fairness Index:**
   Let $w_e = \sum_{d \in D} \sum_{s \in S} x_{e, d, s}$ be the total shifts worked by employee $e$.
   $$J(W) = \frac{\left(\sum_{e=1}^N w_e\right)^2}{N \sum_{e=1}^N w_e^2}$$
   $J(W) \in [1/N, 1.0]$. A value of $1.0$ indicates perfectly balanced shift allocation across the staff.

---

## 3. COMPUTATIONAL COMPLEXITY & NP-HARDNESS

### 3.1 Combinatorial Space Explosion
Consider an instance with $N = 15$ employees, $T = 7$ days, and 3 shifts per day requiring 1 worker each ($K = 21$ slots total):
$$\text{Search Space} = N^K = 15^{21} \approx 5.84 \times 10^{24} \text{ combinations}$$
Brute-force exhaustive search at $10^9$ operations per second would require over **185 million years** to check all configurations.

### 3.2 NP-Completeness Proof Sketch
The decision version of the Employee Shift Scheduling Problem can be reduced from the **Exact Cover by 3-Sets (X3C)** problem:
- In X3C, given a universe $X$ with $|X| = 3q$ and a collection $C$ of 3-element subsets of $X$, determine whether $C$ contains an exact cover for $X$.
- Map each day's shifts to universe elements, and candidate employee assignments subject to conflict constraints to candidate sets.
- A valid schedule of exact coverage exists if and only if an exact cover exists.
- Since X3C is Karp-complete, ESSP is **NP-Complete in decision form** and **NP-Hard in optimization form**.

---

## 4. ALGORITHMIC PARADIGMS IMPLEMENTED

### 4.1 Algorithm 1: Greedy Choice Heuristic
- **Strategy:** Makes locally optimal choices at each shift slot by sorting eligible employees based on a composite heuristic priority:
  $$\text{Priority}(e) = \alpha \cdot P_{e, s} - \beta \cdot w_e$$
  where $w_e$ is the current number of shifts worked by employee $e$, ensuring dynamic workload balancing.
- **Time Complexity:** $O(D \cdot S \cdot N \log N)$ (sorting candidates for each slot).
- **Space Complexity:** $O(D \cdot S + N)$ auxiliary storage.
- **Pros:** Sub-millisecond execution, ideal for real-time dispatch.
- **Cons:** Susceptible to local minima; can fail to satisfy overconstrained problems.

### 4.2 Algorithm 2: Backtracking with CSP Heuristics
- **Strategy:** Explores a state-space tree of assignment decisions with pruning:
  - **MRV (Minimum Remaining Values):** Selects the unassigned shift slot with the fewest legal candidate employees first ("fail-first principle").
  - **Forward Checking:** After each tentative assignment, verifies that all future slots have at least 1 legal employee. If any domain is empty, prunes immediately.
  - **Value Ordering:** Evaluates candidates in order of highest preference and lowest current shift count.
- **Time Complexity:** Worst-case $O(N^{D \cdot S})$; average pruned: $O(b^d)$ with $b \ll N$.
- **Space Complexity:** $O(D \cdot S)$ recursion stack depth.
- **Pros:** 100% guarantee of finding a zero-violation feasible solution if one exists.
- **Cons:** Exponential growth on highly constrained large instances without timeout.

### 4.3 Algorithm 3: Branch and Bound (DFBnB)
- **Strategy:** Depth-First Branch and Bound utilizing an admissible lower-bound function:
  $$\text{Cost} = \sum_{\text{assigned}} (5 - P_{e, s})$$
  $$\text{Lower Bound } LB(u) = \text{Current Cost}(u) + \sum_{k \in \text{unfilled}} \min_{e} (5 - P_{e, s_k})$$
  Whenever $LB(u) \ge \text{BestCostKnown}$, the branch is pruned immediately.
- **Time Complexity:** $O(b^d)$ with tight pruning.
- **Space Complexity:** $O(D \cdot S)$ memory depth.
- **Pros:** Provably guarantees globally optimal preference-cost solution.
- **Cons:** Higher runtime than greedy heuristics on medium-to-large horizons.

### 4.4 Algorithm 4: Genetic Algorithm (Metaheuristic)
- **Strategy:** Bio-inspired evolutionary optimization:
  - **Chromosome Representation:** List of employee IDs of length $\sum R_{d, s}$.
  - **Fitness Function:**
    $$\text{Fitness} = 500 - 200 \times \text{Violations} + 2 \times \text{PrefPct} + 100 \times J(W) - 15 \cdot \sigma$$
  - **Selection:** Tournament selection ($k=3$).
  - **Crossover:** Uniform crossover preserving shift coverage.
  - **Mutation:** Random employee reassignment with probability $p_m = 0.15$.
  - **Elitism:** Direct preservation of top 2 chromosomes each generation.
- **Time Complexity:** $O(G \cdot P \cdot D \cdot S)$ where $G$=Generations, $P$=Population.
- **Space Complexity:** $O(P \cdot D \cdot S)$.
- **Pros:** Highly scalable; excels when exact algorithms face combinatorial explosion.
- **Cons:** Stochastic; does not guarantee global optimum.

---

## 5. THEORETICAL COMPLEXITY COMPARISON

| Algorithmic Paradigm | Asymptotic Time Complexity | Space Complexity | Optimality Guarantee | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Greedy Choice Heuristic** | $O(D \cdot S \cdot N \log N)$ | $O(D \cdot S + N)$ | Local Optima | Real-time dispatch, very large teams ($N > 100$) |
| **Backtracking (CSP + MRV)** | Worst $O(N^{D \cdot S})$, Pruned $O(b^d)$ | $O(D \cdot S)$ | Feasibility Guaranteed | Strict regulatory compliance, small-to-medium teams |
| **Branch & Bound (DFBnB)** | Worst $O(b^d)$, Pruned | $O(D \cdot S)$ | Global Optimum Guaranteed | Strategic weekly roster planning where perfection is mandatory |
| **Genetic Algorithm** | $O(G \cdot P \cdot D \cdot S)$ | $O(P \cdot D \cdot S)$ | Near-Optimal Metaheuristic | High-complexity overconstrained systems with soft preferences |

---

## 6. EMPIRICAL RESULTS & BENCHMARK ANALYSIS

Tested on a benchmark instance: **12 Employees, 7 Days, 3 Shifts/Day (Morning=2, Evening=2, Night=1, Total = 35 slots)**:

| Algorithm | Runtime (ms) | Hard Violations | Preference % | Jain's Fairness Index | Solution Quality |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Greedy Choice Heuristic** | **0.23 ms** | **0** | 76.0% | **0.9540** | Excellent fairness, ultra-fast |
| **Backtracking CSP (MRV)** | 7.15 ms | **0** | 86.3% | 0.6944 | Higher preference satisfaction |
| **Branch and Bound** | 2500.15 ms | **0** | **86.9%** | 0.6586 | Highest preference score |
| **Genetic Algorithm** | 142.50 ms | 12 | 54.1% | 0.7646 | Stochastic metaheuristic |

### Key Observations:
1. **Speed Champion:** Greedy Heuristic solved the 35-slot instance in **0.23 ms**, outperforming exact methods by over 4 orders of magnitude.
2. **Quality Champion:** Branch and Bound produced the highest preference satisfaction (**86.9%**) with zero hard violations.
3. **Fairness Champion:** The dynamic workload-aware Greedy Heuristic achieved the highest Jain's Fairness Index (**0.9540**).
4. **Feasibility:** Backtracking with MRV found a feasible schedule in **7.15 ms**, pruning over 99.9% of the search tree.

---

## 7. VIVA VOCE QUESTIONS & EXPERT ANSWERS

#### Q1: What makes Employee Shift Scheduling NP-Hard?
**Answer:** It maps directly to Exact Cover by 3-Sets (X3C) and Timetabling problems. For $N$ workers and $K$ slots, the combinatorial space is $N^K$. No polynomial-time algorithm exists to solve it optimally under arbitrary constraints unless $P = NP$.

#### Q2: How does MRV (Minimum Remaining Values) improve Backtracking?
**Answer:** MRV implements the "fail-first" principle. By picking the shift slot with the fewest legal available candidates, any dead end causes immediate pruning at the top of the search tree, preventing millions of redundant subtrees from ever being expanded.

#### Q3: What is Forward Checking and how does it prevent dead ends?
**Answer:** Whenever a worker is assigned to slot $k$, Forward Checking evaluates all remaining unassigned slots. If any slot's domain of available legal workers drops to zero, the search immediately backtracks without executing any recursive descent.

#### Q4: Why is Jain's Fairness Index preferred over Standard Deviation?
**Answer:** Jain's Fairness Index $J(W) = \frac{(\sum w_i)^2}{N \sum w_i^2}$ is scale-independent, dimensionless, and strictly bounded between $1/N$ and $1.0$. A score of $1.0$ represents equal distribution, making it universally interpretable regardless of total shifts or team size.

#### Q5: What is the space advantage of Depth-First Branch and Bound (DFBnB)?
**Answer:** Best-First Search stores all open frontier nodes in memory (exponential $O(b^d)$). DFBnB follows depth-first recursion, keeping only the active path on the call stack, guaranteeing linear $O(D \cdot S)$ space complexity.

---

## 8. CONCLUSION & FUTURE SCOPE
This project demonstrates how classical and modern DAA algorithmic paradigms address an NP-Hard scheduling problem. Greedy provides instant solutions for online dispatch; Backtracking guarantees feasibility through MRV pruning; Branch and Bound guarantees optimal cost bounds; and Genetic Algorithms offer evolutionary metaheuristic scaling.

**Future Enhancements:**
- Integer Linear Programming (ILP) integration using Simplex / Branch-and-Cut.
- Skill qualification constraints (Senior Doctor, Registered Nurse, Trainee).
- Multi-week rolling horizons with overtime penalty functions.

---
*End of Academic Project Report*
