import numpy as np
from typing import List, Optional, Tuple


class NormalFormGame:
    def __init__(
        self,
        payoff_1: np.ndarray,
        payoff_2: np.ndarray,
        game_name: str = "Normal Form Game",
        player1_name: str = "Player 1",
        player2_name: str = "Player 2",
        strategy_names_1: Optional[List[str]] = None,
        strategy_names_2: Optional[List[str]] = None,
    ):
        self.payoff_1 = np.array(payoff_1, dtype=float)
        self.payoff_2 = np.array(payoff_2, dtype=float)

        if self.payoff_1.shape != self.payoff_2.shape:
            raise ValueError(
                f"Payoff matrices must share identical shapes. "
                f"Got {self.payoff_1.shape} vs {self.payoff_2.shape}."
            )
        if self.payoff_1.ndim != 2:
            raise ValueError("Payoff matrices must be 2-dimensional.")

        self.m, self.n = self.payoff_1.shape
        self.game_name = game_name
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.strategy_names_1 = (
            strategy_names_1 if strategy_names_1 is not None
            else [f"S{i + 1}" for i in range(self.m)]
        )
        self.strategy_names_2 = (
            strategy_names_2 if strategy_names_2 is not None
            else [f"S{j + 1}" for j in range(self.n)]
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.m, self.n

    def payoffs_at(self, i: int, j: int) -> Tuple[float, float]:
        return float(self.payoff_1[i, j]), float(self.payoff_2[i, j])

    def is_zero_sum(self) -> bool:
        return bool(np.allclose(self.payoff_1 + self.payoff_2, 0.0))

    def is_constant_sum(self) -> bool:
        sums = self.payoff_1 + self.payoff_2
        return bool(np.allclose(sums, sums[0, 0]))

    def is_symmetric(self) -> bool:
        return bool(
            self.m == self.n
            and np.allclose(self.payoff_1, self.payoff_2.T)
        )

    def best_response_p1_pure(self, j: int) -> List[int]:
        col = self.payoff_1[:, j]
        max_val = np.max(col)
        return [i for i in range(self.m) if col[i] >= max_val - 1e-10]

    def best_response_p2_pure(self, i: int) -> List[int]:
        row = self.payoff_2[i, :]
        max_val = np.max(row)
        return [j for j in range(self.n) if row[j] >= max_val - 1e-10]

    def br_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        br1 = np.zeros((self.m, self.n), dtype=bool)
        br2 = np.zeros((self.m, self.n), dtype=bool)
        for j in range(self.n):
            for i in self.best_response_p1_pure(j):
                br1[i, j] = True
        for i in range(self.m):
            for j in self.best_response_p2_pure(i):
                br2[i, j] = True
        return br1, br2

    def pure_nash_matrix(self) -> np.ndarray:
        br1, br2 = self.br_matrix()
        return br1 & br2
