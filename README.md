# Nash Equilibrium Solver

A Python implementation of Nash Equilibrium computation for finite two-player normal-form games. Built from first principles using the **support enumeration algorithm**, **linear programming for dominance testing**, and the **Von Neumann minimax theorem** for zero-sum games. This project has been coded from first principles and is intended for educational purposes.

---

## Mathematical Foundation

### Nash Equilibrium (Nash, 1950)

A mixed strategy profile $(\sigma_1^*, \sigma_2^*)$ is a Nash Equilibrium if and only if neither player can profitably deviate:

$$u_i(\sigma_i^*, \sigma_{-i}^*) \geq u_i(\sigma_i, \sigma_{-i}^*) \quad \forall \sigma_i \in \Delta(S_i), \; i \in \{1, 2\}$$

### Support Enumeration Algorithm

For each pair of supports $(S_1 \subseteq \mathcal{S}_1, \; S_2 \subseteq \mathcal{S}_2)$:

**Step 1 — Indifference conditions.** Find $q^*$ supported on $S_2$ such that Player 1 is indifferent over all strategies in $S_1$:

$$\sum_{j \in S_2} q_j^* \cdot A_{ij} = \sum_{j \in S_2} q_j^* \cdot A_{i'j} \quad \forall \, i, i' \in S_1$$

This gives the linear system:

$$\begin{pmatrix} A_{S_1, S_2}^{(1)} - A_{S_1, S_2}^{(0)} \\ \mathbf{1}^\top \end{pmatrix} \mathbf{q}_{S_2} = \begin{pmatrix} \mathbf{0} \\ 1 \end{pmatrix}$$

**Step 2** — Repeat for Player 2 to find $p^*$ supported on $S_1$.

**Step 3 — Verification.** Confirm no profitable deviation exists outside the support:

$$A_{ij} \cdot q^* \leq v_1^* \quad \forall \, i \notin S_1, \qquad p^{*\top} B_{\cdot j} \leq v_2^* \quad \forall \, j \notin S_2$$

### LP-Based Strict Dominance (Pearce, 1984)

Strategy $s_i$ is strictly dominated if:

$$\exists \, \sigma_i \in \Delta(\mathcal{S}_i \setminus \{s_i\}): \; u_i(\sigma_i, s_{-i}) > u_i(s_i, s_{-i}) \quad \forall \, s_{-i}$$

Computed via linear program:

$$\max \; v \quad \text{s.t.} \quad \sum_{k \neq i} p_k A_{kj} - A_{ij} \geq v \; \forall j, \quad \sum_k p_k = 1, \quad p \geq 0$$

Strategy $s_i$ is strictly dominated $\iff v^* > 0$.

### Minimax Theorem (Von Neumann, 1928)

For zero-sum games, the game value $v^*$ satisfies:

$$v^* = \max_p \min_q \; p^\top A q = \min_q \max_p \; p^\top A q$$

Computed directly via LP using the HiGHS solver.

---

## Features

| Feature | Description |
|---|---|
| **Support Enumeration** | Finds all Nash Equilibria, pure and mixed, using exhaustive support search |
| **LP Dominance Testing** | Strict and weak dominance using HiGHS linear program solver |
| **IESDS** | Iterated Elimination of Strictly Dominated Strategies with full elimination history |
| **Pareto Analysis** | Identifies Pareto-optimal outcomes and social welfare maximiser |
| **Minimax Values** | Game values for zero-sum games using Von Neumann's theorem |
| **NE vs Social Optimum** | Social welfare loss at each Nash equilibrium relative to the optimum |
| **Classic Game Library** | 8 canonical games with named strategies |
| **Custom Game Input** | Interactive matrix entry or JSON import |
| **Rich Terminal UI** | Colour-coded payoff matrix with best-response and Nash highlighting |
| **JSON Export** | Full results exportable to JSON |

---

## Installation

```bash
git clone https://github.com/yourusername/nash-solver.git
cd nash-solver
pip install -r requirements.txt
python main.py
```

---

## Usage

### Interactive CLI

```bash
python main.py
```

The terminal menu guides you through:
- Selecting a classic game or entering a custom one
- Full analysis output (matrix, equilibria, IESDS, Pareto, dominance)
- Sub-menu for targeted re-analysis and export

### Programmatic API

```python
import numpy as np
from game import NormalFormGame
from solvers import NashSolver
from analysis import GameAnalyzer

A = np.array([[2, 0], [0, 1]], dtype=float)  # Player 1 payoffs (Battle of Sexes)
B = np.array([[1, 0], [0, 2]], dtype=float)  # Player 2 payoffs

game = NormalFormGame(A, B,
    game_name="Battle of the Sexes",
    strategy_names_1=["Opera", "Football"],
    strategy_names_2=["Opera", "Football"])

solver  = NashSolver(game)
analyzer = GameAnalyzer(game)

equilibria = solver.find_all_nash()        # All NE via support enumeration
avail1, avail2, hist = analyzer.iesds()   # IESDS with elimination history
pareto = analyzer.pareto_optimal_outcomes()
dom    = analyzer.dominance_analysis()
```

### JSON Game Format

```json
{
  "game_name": "My Game",
  "player1_name": "Player 1",
  "player2_name": "Player 2",
  "strategy_names_1": ["Top", "Bottom"],
  "strategy_names_2": ["Left", "Right"],
  "payoff_1": [[3, 0], [0, 2]],
  "payoff_2": [[3, 0], [0, 2]]
}
```

---

## Classic Games Included

| Game | # NE | Type | Key Result |
|---|---|---|---|
| Prisoner's Dilemma | 1 | Pure | Unique NE is Pareto-suboptimal (social dilemma) |
| Battle of the Sexes | 3 | Pure × 2, Mixed × 1 | Multiple equilibria, coordination problem |
| Stag Hunt | 3 | Pure × 2, Mixed × 1 | Risk-dominant vs payoff-dominant equilibrium |
| Hawk-Dove | 3 | Pure × 2, Mixed × 1 | Asymmetric pure NE; mixed is Pareto-dominated |
| Matching Pennies | 1 | Mixed | Zero-sum; unique NE at (½, ½) |
| Rock-Paper-Scissors | 1 | Mixed | Zero-sum; unique NE at (⅓, ⅓, ⅓) |
| Coordination Game | 3 | Pure × 2, Mixed × 1 | Technology adoption / standard-setting |
| Cournot Duopoly | 1 | Pure | Discretised quantity competition |

---

## Project Structure

```
nash_solver/
├── game.py          NormalFormGame class — payoff matrices, BR computation
├── solvers.py       NashEquilibrium dataclass · NashSolver (support enumeration)
├── analysis.py      GameAnalyzer — LP dominance, IESDS, Pareto, minimax
├── display.py       Rich terminal UI — colour-coded matrix, panels, tables
├── examples.py      8 classic games with named strategies
├── main.py          Interactive CLI entry point
└── requirements.txt numpy · scipy · rich
```

---

## References

- Nash, J. (1950). *Equilibrium points in n-person games.* PNAS, 36(1), 48–49.
- Von Neumann, J. (1928). *Zur Theorie der Gesellschaftsspiele.* Mathematische Annalen, 100(1), 295–320.
- Pearce, D. G. (1984). *Rationalizable strategic behavior and the problem of perfection.* Econometrica, 52(4), 1029–1050.
- Osborne, M. J., & Rubinstein, A. (1994). *A Course in Game Theory.* MIT Press.