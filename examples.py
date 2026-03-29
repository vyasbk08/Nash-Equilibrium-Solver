import numpy as np
from game import NormalFormGame


def prisoners_dilemma() -> NormalFormGame:
    return NormalFormGame(
        payoff_1=np.array([[-1, -3], [0, -2]], dtype=float),
        payoff_2=np.array([[-1, 0], [-3, -2]], dtype=float),
        game_name="Prisoner's Dilemma",
        player1_name="Prisoner 1",
        player2_name="Prisoner 2",
        strategy_names_1=["Cooperate", "Defect"],
        strategy_names_2=["Cooperate", "Defect"],
    )


def battle_of_sexes() -> NormalFormGame:
    return NormalFormGame(
        payoff_1=np.array([[2, 0], [0, 1]], dtype=float),
        payoff_2=np.array([[1, 0], [0, 2]], dtype=float),
        game_name="Battle of the Sexes",
        player1_name="Player 1",
        player2_name="Player 2",
        strategy_names_1=["Opera", "Football"],
        strategy_names_2=["Opera", "Football"],
    )


def stag_hunt() -> NormalFormGame:
    return NormalFormGame(
        payoff_1=np.array([[4, 0], [3, 3]], dtype=float),
        payoff_2=np.array([[4, 3], [0, 3]], dtype=float),
        game_name="Stag Hunt",
        player1_name="Hunter 1",
        player2_name="Hunter 2",
        strategy_names_1=["Stag", "Hare"],
        strategy_names_2=["Stag", "Hare"],
    )


def hawk_dove() -> NormalFormGame:
    return NormalFormGame(
        payoff_1=np.array([[-1, 4], [0, 2]], dtype=float),
        payoff_2=np.array([[-1, 0], [4, 2]], dtype=float),
        game_name="Hawk-Dove (Chicken)",
        player1_name="Player 1",
        player2_name="Player 2",
        strategy_names_1=["Hawk", "Dove"],
        strategy_names_2=["Hawk", "Dove"],
    )


def matching_pennies() -> NormalFormGame:
    return NormalFormGame(
        payoff_1=np.array([[1, -1], [-1, 1]], dtype=float),
        payoff_2=np.array([[-1, 1], [1, -1]], dtype=float),
        game_name="Matching Pennies",
        player1_name="Matcher",
        player2_name="Mismatcher",
        strategy_names_1=["Heads", "Tails"],
        strategy_names_2=["Heads", "Tails"],
    )


def rock_paper_scissors() -> NormalFormGame:
    return NormalFormGame(
        payoff_1=np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float),
        payoff_2=np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float),
        game_name="Rock-Paper-Scissors",
        player1_name="Player 1",
        player2_name="Player 2",
        strategy_names_1=["Rock", "Paper", "Scissors"],
        strategy_names_2=["Rock", "Paper", "Scissors"],
    )


def coordination_game() -> NormalFormGame:
    return NormalFormGame(
        payoff_1=np.array([[3, 0], [0, 2]], dtype=float),
        payoff_2=np.array([[3, 0], [0, 2]], dtype=float),
        game_name="Coordination Game",
        player1_name="Firm A",
        player2_name="Firm B",
        strategy_names_1=["Technology A", "Technology B"],
        strategy_names_2=["Technology A", "Technology B"],
    )


def cournot_duopoly() -> NormalFormGame:
    return NormalFormGame(
        payoff_1=np.array([[0, 8, 4], [12, 8, 4], [8, 4, 0]], dtype=float),
        payoff_2=np.array([[0, 12, 8], [8, 8, 4], [4, 4, 0]], dtype=float),
        game_name="Cournot Duopoly (Discretised)",
        player1_name="Firm 1",
        player2_name="Firm 2",
        strategy_names_1=["Low (q=1)", "Medium (q=2)", "High (q=3)"],
        strategy_names_2=["Low (q=1)", "Medium (q=2)", "High (q=3)"],
    )


EXAMPLE_GAMES = {
    "1": ("Prisoner's Dilemma",           prisoners_dilemma),
    "2": ("Battle of the Sexes",           battle_of_sexes),
    "3": ("Stag Hunt",                     stag_hunt),
    "4": ("Hawk-Dove (Chicken)",            hawk_dove),
    "5": ("Matching Pennies",               matching_pennies),
    "6": ("Rock-Paper-Scissors",            rock_paper_scissors),
    "7": ("Coordination Game",              coordination_game),
    "8": ("Cournot Duopoly (Discretised)",  cournot_duopoly),
}
