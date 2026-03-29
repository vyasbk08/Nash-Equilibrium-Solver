# Nash Equilibrium Solver

A Python implementation of Nash Equilibrium computation for finite two-player normal-form games. Built from first principles using the **support enumeration algorithm**, **linear programming for dominance testing**, and the **Von Neumann minimax theorem** for zero-sum games. This project has been coded from first principles and is intended for educational purposes.

---

## Mathematical Foundation

### Nash Equilibrium (Nash, 1950)

A mixed strategy profile $(\sigma_{1}^{\ast}, \sigma_{2}^{\ast})$ is a Nash Equilibrium if and only if neither player can profitably deviate:

$$u_{i}(\sigma_{i}^{\ast}, \sigma_{-i}^{\ast}) \geq u_{i}(\sigma_{i}, \sigma_{-i}^{\ast}) \quad \forall \sigma_{i} \in \Delta(S_{i}), \; i \in \{1, 2\}$$

### Support Enumeration Algorithm

For each pair of supports $(S_{1} \subseteq \mathcal{S}_{1}, \; S_{2} \subseteq \mathcal{S}_{2})$:

**Step 1 — Indifference conditions.** Find $q^{\ast}$ supported on $S_{2}$ such that Player 1 is indifferent over all strategies in $S_{1}$:

$$\sum_{j \in S_{2}} q_{j}^{\ast} \cdot A_{ij} = \sum_{j \in S_{2}} q_{j}^{\ast} \cdot A_{i'j} \quad \forall \, i, i' \in S_{1}$$

This gives the linear system:

$$\begin{pmatrix} A_{S_{1}, S_{2}}^{(1)} - A_{S_{1}, S_{2}}^{(0)} \\ \mathbf{1}^\top \end{pmatrix} \mathbf{q}_{S_{2}} = \begin{pmatrix} \mathbf{0} \\ 1 \end{pmatrix}$$

**Step 2** — Repeat for Player 2 to find $p^{\ast}$ supported on $S_{1}$.

**Step 3 — Verification.** Confirm no profitable deviation exists outside the support:

$$A_{ij} \cdot q^{\ast} \leq v_{1}^{\ast} \quad \forall \, i \notin S_{1}, \qquad p^{\ast\top} B_{\cdot j} \leq v_{2}^{\ast} \quad \forall \, j \notin S_{2}$$

### LP-Based Strict Dominance (Pearce, 1984)

Strategy $s_{i}$ is strictly dominated if:

$$\exists \, \sigma_{i} \in \Delta(\mathcal{S}_{i} \setminus \{s_{i}\}): \; u_{i}(\sigma_{i}, s_{-i}) > u_{i}(s_{i}, s_{-i}) \quad \forall \, s_{-i}$$

Computed via linear program:

$$\max \; v \quad \text{s.t.} \quad \sum_{k \neq i} p_{k} A_{kj} - A_{ij} \geq v \; \forall j, \quad \sum_{k} p_{k} = 1, \quad p \geq 0$$

Strategy $s_{i}$ is strictly dominated $\iff v^{\ast} > 0$.

### Minimax Theorem (Von Neumann, 1928)

For zero-sum games, the game value $v^{\ast}$ satisfies:

$$v^{\ast} = \max_{p} \min_{q} \; p^\top A q = \min_{q} \max_{p} \; p^\top A q$$

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
git clone [https://github.com/yourusername/nash-solver.git](https://github.com/yourusername/nash-solver.git)
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
