import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Optional, Tuple

from game import NormalFormGame
from solvers import NashEquilibrium


class GameAnalyzer:
    TOLERANCE = 1e-9

    def __init__(self, game: NormalFormGame):
        self.game = game
        self.A = game.payoff_1
        self.B = game.payoff_2
        self.m, self.n = game.shape

    def best_response_p1_mixed(self, q: np.ndarray) -> List[int]:
        payoffs = self.A @ q
        max_payoff = float(np.max(payoffs))
        return [i for i in range(self.m) if payoffs[i] >= max_payoff - self.TOLERANCE]

    def best_response_p2_mixed(self, p: np.ndarray) -> List[int]:
        payoffs = self.B.T @ p
        max_payoff = float(np.max(payoffs))
        return [j for j in range(self.n) if payoffs[j] >= max_payoff - self.TOLERANCE]

    def is_strictly_dominated_p1(
        self,
        strategy_idx: int,
        avail_1: List[int],
        avail_2: List[int],
    ) -> bool:
        others = [i for i in avail_1 if i != strategy_idx]
        if not others:
            return False

        n_others = len(others)
        n_cols = len(avail_2)
        A_sub = self.A[np.ix_(others, avail_2)]
        dominated_row = self.A[np.ix_([strategy_idx], avail_2)][0]

        c = np.zeros(n_others + 1)
        c[-1] = -1.0

        A_ub = np.zeros((n_cols, n_others + 1))
        A_ub[:, :n_others] = -A_sub.T
        A_ub[:, -1] = 1.0
        b_ub = -dominated_row

        A_eq = np.zeros((1, n_others + 1))
        A_eq[0, :n_others] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * n_others + [(None, None)]

        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs"
        )
        return result.status == 0 and (-result.fun) > self.TOLERANCE

    def is_strictly_dominated_p2(
        self,
        strategy_idx: int,
        avail_1: List[int],
        avail_2: List[int],
    ) -> bool:
        others = [j for j in avail_2 if j != strategy_idx]
        if not others:
            return False

        n_others = len(others)
        n_rows = len(avail_1)
        B_sub = self.B[np.ix_(avail_1, others)]
        dominated_col = self.B[np.ix_(avail_1, [strategy_idx])][:, 0]

        c = np.zeros(n_others + 1)
        c[-1] = -1.0

        A_ub = np.zeros((n_rows, n_others + 1))
        A_ub[:, :n_others] = -B_sub
        A_ub[:, -1] = 1.0
        b_ub = -dominated_col

        A_eq = np.zeros((1, n_others + 1))
        A_eq[0, :n_others] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * n_others + [(None, None)]

        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs"
        )
        return result.status == 0 and (-result.fun) > self.TOLERANCE

    def is_weakly_dominated_p1(
        self,
        strategy_idx: int,
        avail_1: List[int],
        avail_2: List[int],
    ) -> bool:
        others = [i for i in avail_1 if i != strategy_idx]
        if not others:
            return False
        A_others = self.A[np.ix_(others, avail_2)]
        dominated_row = self.A[np.ix_([strategy_idx], avail_2)][0]
        for other_i_local, other_i in enumerate(others):
            row = A_others[other_i_local]
            if np.all(row >= dominated_row - self.TOLERANCE) and np.any(row > dominated_row + self.TOLERANCE):
                return True
        return False

    def is_weakly_dominated_p2(
        self,
        strategy_idx: int,
        avail_1: List[int],
        avail_2: List[int],
    ) -> bool:
        others = [j for j in avail_2 if j != strategy_idx]
        if not others:
            return False
        B_others = self.B[np.ix_(avail_1, others)]
        dominated_col = self.B[np.ix_(avail_1, [strategy_idx])][:, 0]
        for other_j_local in range(len(others)):
            col = B_others[:, other_j_local]
            if np.all(col >= dominated_col - self.TOLERANCE) and np.any(col > dominated_col + self.TOLERANCE):
                return True
        return False

    def iesds(self) -> Tuple[List[int], List[int], List[str]]:
        avail_1 = list(range(self.m))
        avail_2 = list(range(self.n))
        history: List[str] = []
        changed = True

        while changed:
            changed = False

            to_remove_1 = [
                i for i in avail_1
                if self.is_strictly_dominated_p1(i, avail_1, avail_2)
            ]
            for i in to_remove_1:
                avail_1.remove(i)
                history.append(
                    f"{self.game.player1_name} strategy "
                    f"'{self.game.strategy_names_1[i]}' eliminated (strictly dominated)"
                )
                changed = True

            to_remove_2 = [
                j for j in avail_2
                if self.is_strictly_dominated_p2(j, avail_1, avail_2)
            ]
            for j in to_remove_2:
                avail_2.remove(j)
                history.append(
                    f"{self.game.player2_name} strategy "
                    f"'{self.game.strategy_names_2[j]}' eliminated (strictly dominated)"
                )
                changed = True

        return avail_1, avail_2, history

    def dominance_analysis(self) -> Dict[str, List[int]]:
        avail_1 = list(range(self.m))
        avail_2 = list(range(self.n))
        return {
            "strictly_dominated_p1": [
                i for i in range(self.m)
                if self.is_strictly_dominated_p1(i, avail_1, avail_2)
            ],
            "strictly_dominated_p2": [
                j for j in range(self.n)
                if self.is_strictly_dominated_p2(j, avail_1, avail_2)
            ],
            "weakly_dominated_p1": [
                i for i in range(self.m)
                if (not self.is_strictly_dominated_p1(i, avail_1, avail_2)
                    and self.is_weakly_dominated_p1(i, avail_1, avail_2))
            ],
            "weakly_dominated_p2": [
                j for j in range(self.n)
                if (not self.is_strictly_dominated_p2(j, avail_1, avail_2)
                    and self.is_weakly_dominated_p2(j, avail_1, avail_2))
            ],
        }

    def _is_pareto_dominated(self, i: int, j: int) -> bool:
        u1, u2 = float(self.A[i, j]), float(self.B[i, j])
        for ii in range(self.m):
            for jj in range(self.n):
                if ii == i and jj == j:
                    continue
                v1, v2 = float(self.A[ii, jj]), float(self.B[ii, jj])
                weakly_better = (
                    v1 >= u1 - self.TOLERANCE and v2 >= u2 - self.TOLERANCE
                )
                strictly_better = (
                    v1 > u1 + self.TOLERANCE or v2 > u2 + self.TOLERANCE
                )
                if weakly_better and strictly_better:
                    return True
        return False

    def pareto_optimal_outcomes(self) -> List[Tuple[int, int]]:
        return [
            (i, j)
            for i in range(self.m)
            for j in range(self.n)
            if not self._is_pareto_dominated(i, j)
        ]

    def social_welfare_maximizer(self) -> Tuple[int, int, float]:
        best_i, best_j = 0, 0
        best_sw = float("-inf")
        for i in range(self.m):
            for j in range(self.n):
                sw = float(self.A[i, j]) + float(self.B[i, j])
                if sw > best_sw:
                    best_sw = sw
                    best_i, best_j = i, j
        return best_i, best_j, best_sw

    def minimax_value_p1(self) -> float:
        n_vars = self.m + 1
        c = np.zeros(n_vars)
        c[-1] = -1.0

        A_ub = np.zeros((self.n, n_vars))
        A_ub[:, : self.m] = -self.A.T
        A_ub[:, -1] = 1.0
        b_ub = np.zeros(self.n)

        A_eq = np.zeros((1, n_vars))
        A_eq[0, : self.m] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * self.m + [(None, None)]
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs"
        )
        return float(-result.fun) if result.status == 0 else float("nan")

    def minimax_value_p2(self) -> float:
        n_vars = self.n + 1
        c = np.zeros(n_vars)
        c[-1] = -1.0

        A_ub = np.zeros((self.m, n_vars))
        A_ub[:, : self.n] = -self.B
        A_ub[:, -1] = 1.0
        b_ub = np.zeros(self.m)

        A_eq = np.zeros((1, n_vars))
        A_eq[0, : self.n] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, None)] * self.n + [(None, None)]
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs"
        )
        return float(-result.fun) if result.status == 0 else float("nan")

    def nash_relative_to_pareto(
        self,
        equilibria: List[NashEquilibrium],
    ) -> List[Dict]:
        pareto = set(self.pareto_optimal_outcomes())
        sw_i, sw_j, sw_max = self.social_welfare_maximizer()
        analysis = []
        for ne in equilibria:
            if ne.is_pure:
                i, j = ne.support_p1[0], ne.support_p2[0]
                is_pareto = (i, j) in pareto
                sw_loss = sw_max - ne.social_welfare()
            else:
                is_pareto = None
                sw_loss = sw_max - ne.social_welfare()
            analysis.append({
                "ne": ne,
                "is_pareto_optimal": is_pareto,
                "social_welfare": ne.social_welfare(),
                "sw_loss_vs_max": sw_loss,
            })
        return analysis
