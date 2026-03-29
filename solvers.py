import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple

from game import NormalFormGame


@dataclass
class NashEquilibrium:
    strategy_p1: np.ndarray
    strategy_p2: np.ndarray
    payoff_p1: float
    payoff_p2: float
    is_pure: bool
    support_p1: List[int]
    support_p2: List[int]

    def social_welfare(self) -> float:
        return self.payoff_p1 + self.payoff_p2

    def is_fully_mixed(self) -> bool:
        return len(self.support_p1) > 1 and len(self.support_p2) > 1

    def equilibrium_type(self) -> str:
        if self.is_pure:
            return "Pure Strategy NE"
        if self.is_fully_mixed():
            return "Fully Mixed NE"
        return "Mixed Strategy NE"

    def pareto_comparable(self, other: "NashEquilibrium") -> Optional[str]:
        if (self.payoff_p1 >= other.payoff_p1 and self.payoff_p2 >= other.payoff_p2
                and (self.payoff_p1 > other.payoff_p1 or self.payoff_p2 > other.payoff_p2)):
            return "self dominates other"
        if (other.payoff_p1 >= self.payoff_p1 and other.payoff_p2 >= self.payoff_p2
                and (other.payoff_p1 > self.payoff_p1 or other.payoff_p2 > self.payoff_p2)):
            return "other dominates self"
        return None


class NashSolver:
    TOLERANCE = 1e-9

    def __init__(self, game: NormalFormGame):
        self.game = game
        self.A = game.payoff_1
        self.B = game.payoff_2
        self.m, self.n = game.shape

    def find_all_nash(self) -> List[NashEquilibrium]:
        return self._support_enumeration()

    def find_pure_nash(self) -> List[NashEquilibrium]:
        results = []
        for i in range(self.m):
            for j in range(self.n):
                if self._is_pure_nash(i, j):
                    p1 = np.zeros(self.m)
                    p2 = np.zeros(self.n)
                    p1[i] = 1.0
                    p2[j] = 1.0
                    results.append(NashEquilibrium(
                        strategy_p1=p1,
                        strategy_p2=p2,
                        payoff_p1=float(self.A[i, j]),
                        payoff_p2=float(self.B[i, j]),
                        is_pure=True,
                        support_p1=[i],
                        support_p2=[j],
                    ))
        return results

    def _is_pure_nash(self, i: int, j: int) -> bool:
        if np.any(self.A[:, j] > self.A[i, j] + self.TOLERANCE):
            return False
        if np.any(self.B[i, :] > self.B[i, j] + self.TOLERANCE):
            return False
        return True

    def _support_enumeration(self) -> List[NashEquilibrium]:
        results: List[NashEquilibrium] = []
        seen: List[Tuple[np.ndarray, np.ndarray]] = []

        for s1_size in range(1, self.m + 1):
            for s2_size in range(1, self.n + 1):
                for s1 in combinations(range(self.m), s1_size):
                    for s2 in combinations(range(self.n), s2_size):
                        p2 = self._solve_indifference(
                            self.A, list(s1), list(s2), self.n
                        )
                        if p2 is None:
                            continue

                        p1 = self._solve_indifference(
                            self.B.T, list(s2), list(s1), self.m
                        )
                        if p1 is None:
                            continue

                        if not self._verify_nash(p1, p2):
                            continue

                        if self._is_duplicate(p1, p2, seen):
                            continue

                        seen.append((p1.copy(), p2.copy()))

                        ne = NashEquilibrium(
                            strategy_p1=p1,
                            strategy_p2=p2,
                            payoff_p1=float(p1 @ self.A @ p2),
                            payoff_p2=float(p1 @ self.B @ p2),
                            is_pure=(s1_size == 1 and s2_size == 1),
                            support_p1=list(s1),
                            support_p2=list(s2),
                        )
                        results.append(ne)

        return results

    def _solve_indifference(
        self,
        M: np.ndarray,
        support_row: List[int],
        support_col: List[int],
        n_total: int,
    ) -> Optional[np.ndarray]:
        M_sub = M[np.ix_(support_row, support_col)]
        nr, nc = M_sub.shape

        if nr == 1 and nc == 1:
            q = np.zeros(n_total)
            q[support_col[0]] = 1.0
            return q

        if nr == 1:
            q_sub = np.ones(nc, dtype=float) / nc
        else:
            eq_rows = [M_sub[k] - M_sub[0] for k in range(1, nr)]
            eq_rows.append(np.ones(nc, dtype=float))
            A_sys = np.array(eq_rows, dtype=float)
            b_sys = np.zeros(nr, dtype=float)
            b_sys[-1] = 1.0

            try:
                if A_sys.shape[0] == A_sys.shape[1]:
                    try:
                        q_sub = np.linalg.solve(A_sys, b_sys)
                    except np.linalg.LinAlgError:
                        return None
                else:
                    q_sub, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                    residual = np.linalg.norm(A_sys @ q_sub - b_sys)
                    if residual > 1e-8:
                        return None
            except Exception:
                return None

        if np.any(q_sub < -self.TOLERANCE):
            return None
        if abs(float(np.sum(q_sub)) - 1.0) > 1e-8:
            return None

        q_sub = np.maximum(q_sub, 0.0)
        q_sub = q_sub / q_sub.sum()

        q = np.zeros(n_total)
        for idx, col in enumerate(support_col):
            q[col] = q_sub[idx]
        return q

    def _verify_nash(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        val_1 = float(p1 @ self.A @ p2)
        val_2 = float(p1 @ self.B @ p2)
        if np.any(self.A @ p2 > val_1 + self.TOLERANCE):
            return False
        if np.any(self.B.T @ p1 > val_2 + self.TOLERANCE):
            return False
        return True

    def _is_duplicate(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        seen: List[Tuple[np.ndarray, np.ndarray]],
        tol: float = 1e-6,
    ) -> bool:
        for sp1, sp2 in seen:
            if np.allclose(p1, sp1, atol=tol) and np.allclose(p2, sp2, atol=tol):
                return True
        return False