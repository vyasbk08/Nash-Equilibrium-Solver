import sys
import json
import os

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.table import Table
from rich import box

from game import NormalFormGame
from solvers import NashSolver
from analysis import GameAnalyzer
from display import (
    console,
    display_full_analysis,
    display_payoff_matrix,
    display_equilibria,
    display_dominance,
    display_iesds,
    display_pareto,
    display_minimax,
    display_ne_vs_pareto,
    _fmt,
)
from examples import EXAMPLE_GAMES


BANNER = """
[bold bright_cyan]
╔══════════════════════════════════════════════════════════════════╗
║          NASH EQUILIBRIUM SOLVER — Normal Form Games           ║
║     Support Enumeration · LP Dominance · Pareto Analysis       ║
╚══════════════════════════════════════════════════════════════════╝
[/][dim]  Built on: NumPy · SciPy (HiGHS LP) · Support Enumeration Algorithm
  Mathematics: Laspeyres Indexing · Von Neumann Minimax · IESDS
[/]"""


def print_banner():
    console.print(BANNER)


def print_menu():
    table = Table(box=box.SIMPLE_HEAVY, border_style="bright_blue", show_header=False, padding=(0, 2))
    table.add_column("Key", style="bold bright_cyan", min_width=5, justify="center")
    table.add_column("Action", style="white", min_width=45)

    table.add_row("[E]", "Select from classic example games")
    table.add_row("[C]", "Enter a custom game manually")
    table.add_row("[J]", "Load a game from JSON file")
    table.add_row("[Q]", "Quit")

    console.print(Panel(table, title="[bold bright_white]Main Menu[/]", border_style="bright_blue"))


def select_example_game() -> NormalFormGame:
    console.print()
    table = Table(box=box.SIMPLE_HEAVY, border_style="cyan", show_header=True, header_style="bold white")
    table.add_column("#", style="bold bright_cyan", justify="center", min_width=4)
    table.add_column("Game", style="white", min_width=35)

    for key, (name, _) in EXAMPLE_GAMES.items():
        table.add_row(key, name)

    console.print(Panel(table, title="[bold bright_cyan]Classic Game Library[/]", border_style="cyan"))
    choice = Prompt.ask(
        "[bold bright_cyan]Select game[/]",
        choices=list(EXAMPLE_GAMES.keys()),
    )
    _, factory = EXAMPLE_GAMES[choice]
    game = factory()
    console.print(f"\n[green]Loaded:[/] [bold]{game.game_name}[/]\n")
    return game


def _parse_matrix(rows: int, cols: int, player_name: str, strategy_names_row: list, strategy_names_col: list) -> np.ndarray:
    console.print(f"\n[bold bright_yellow]Enter payoff matrix for [cyan]{player_name}[/] (row by row):[/]")
    M = np.zeros((rows, cols))
    for i in range(rows):
        while True:
            raw = Prompt.ask(
                f"  Row [bold]{strategy_names_row[i]}[/] "
                f"[dim]({cols} space-separated values)[/]"
            )
            parts = raw.strip().split()
            if len(parts) != cols:
                console.print(f"[red]  Expected {cols} values, got {len(parts)}. Try again.[/]")
                continue
            try:
                values = [float(p) for p in parts]
                M[i] = values
                break
            except ValueError:
                console.print("[red]  Invalid number. Try again.[/]")
    return M


def enter_custom_game() -> NormalFormGame:
    console.print()
    console.print(Rule("[bold bright_yellow]Custom Game Setup[/]", style="bright_yellow"))

    game_name = Prompt.ask("[bold]Game name[/]", default="Custom Game")
    p1_name = Prompt.ask("[bold]Player 1 name[/]", default="Player 1")
    p2_name = Prompt.ask("[bold]Player 2 name[/]", default="Player 2")

    while True:
        try:
            m = int(Prompt.ask(f"[bold]Number of strategies for {p1_name}[/]", default="2"))
            n = int(Prompt.ask(f"[bold]Number of strategies for {p2_name}[/]", default="2"))
            if m < 1 or n < 1:
                raise ValueError
            break
        except ValueError:
            console.print("[red]Please enter positive integers.[/]")

    console.print(f"\n[dim]Enter strategy names for {p1_name}:[/]")
    s1 = [Prompt.ask(f"  Strategy {i + 1} name", default=f"S{i + 1}") for i in range(m)]
    console.print(f"\n[dim]Enter strategy names for {p2_name}:[/]")
    s2 = [Prompt.ask(f"  Strategy {j + 1} name", default=f"S{j + 1}") for j in range(n)]

    A = _parse_matrix(m, n, p1_name, s1, s2)
    B = _parse_matrix(m, n, p2_name, s1, s2)

    return NormalFormGame(
        payoff_1=A,
        payoff_2=B,
        game_name=game_name,
        player1_name=p1_name,
        player2_name=p2_name,
        strategy_names_1=s1,
        strategy_names_2=s2,
    )


def load_from_json() -> NormalFormGame:
    path = Prompt.ask("[bold]Path to JSON file[/]")
    path = path.strip().strip("'\"")
    if not os.path.exists(path):
        console.print(f"[red]File not found: {path}[/]")
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        data = json.load(f)

    required = {"payoff_1", "payoff_2"}
    if not required.issubset(data.keys()):
        raise ValueError(f"JSON must contain keys: {required}")

    return NormalFormGame(
        payoff_1=np.array(data["payoff_1"], dtype=float),
        payoff_2=np.array(data["payoff_2"], dtype=float),
        game_name=data.get("game_name", "Loaded Game"),
        player1_name=data.get("player1_name", "Player 1"),
        player2_name=data.get("player2_name", "Player 2"),
        strategy_names_1=data.get("strategy_names_1", None),
        strategy_names_2=data.get("strategy_names_2", None),
    )


def export_results(game: NormalFormGame, equilibria: list):
    out = {
        "game_name": game.game_name,
        "player1_name": game.player1_name,
        "player2_name": game.player2_name,
        "strategy_names_1": game.strategy_names_1,
        "strategy_names_2": game.strategy_names_2,
        "payoff_1": game.payoff_1.tolist(),
        "payoff_2": game.payoff_2.tolist(),
        "nash_equilibria": [],
    }
    for ne in equilibria:
        out["nash_equilibria"].append({
            "type": ne.equilibrium_type(),
            "strategy_p1": ne.strategy_p1.tolist(),
            "strategy_p2": ne.strategy_p2.tolist(),
            "payoff_p1": ne.payoff_p1,
            "payoff_p2": ne.payoff_p2,
            "social_welfare": ne.social_welfare(),
            "support_p1": ne.support_p1,
            "support_p2": ne.support_p2,
        })

    filename = game.game_name.lower().replace(" ", "_").replace("'", "") + "_results.json"
    with open(filename, "w") as f:
        json.dump(out, f, indent=2)
    console.print(f"\n[green]Results exported to:[/] [bold]{filename}[/]\n")


def post_analysis_menu(game: NormalFormGame, equilibria: list):
    while True:
        console.print()
        table = Table(box=box.SIMPLE_HEAVY, border_style="bright_blue", show_header=False, padding=(0, 2))
        table.add_column("Key", style="bold bright_cyan", min_width=5, justify="center")
        table.add_column("Action", style="white", min_width=45)
        table.add_row("[1]", "View payoff matrix again")
        table.add_row("[2]", "View Nash equilibria")
        table.add_row("[3]", "Run IESDS")
        table.add_row("[4]", "Dominance analysis")
        table.add_row("[5]", "Pareto analysis")
        table.add_row("[6]", "NE vs Social Optimum comparison")
        table.add_row("[7]", "Minimax values (zero-sum games)")
        table.add_row("[X]", "Export results to JSON")
        table.add_row("[M]", "Back to main menu")
        table.add_row("[Q]", "Quit")
        console.print(Panel(table, title="[bold bright_white]Analysis Menu[/]", border_style="bright_blue"))

        choice = Prompt.ask(
            "[bold bright_cyan]Choose[/]",
            choices=["1", "2", "3", "4", "5", "6", "7", "x", "X", "m", "M", "q", "Q"],
        ).upper()

        analyzer = GameAnalyzer(game)

        if choice == "1":
            display_payoff_matrix(game)
        elif choice == "2":
            display_equilibria(equilibria, game)
        elif choice == "3":
            avail_1, avail_2, history = analyzer.iesds()
            display_iesds(avail_1, avail_2, history, game)
        elif choice == "4":
            dom = analyzer.dominance_analysis()
            display_dominance(dom, game)
        elif choice == "5":
            display_pareto(game)
        elif choice == "6":
            display_ne_vs_pareto(equilibria, game)
        elif choice == "7":
            if not game.is_zero_sum():
                console.print("[yellow]Note: This game is not zero-sum. Minimax values are computed independently per player.[/]")
            display_minimax(game)
        elif choice == "X":
            export_results(game, equilibria)
        elif choice == "M":
            break
        elif choice == "Q":
            console.print("\n[bold bright_cyan]Goodbye.[/]\n")
            sys.exit(0)


def main():
    print_banner()

    while True:
        print_menu()
        choice = Prompt.ask(
            "[bold bright_cyan]Choose[/]",
            choices=["e", "E", "c", "C", "j", "J", "q", "Q"],
        ).upper()

        game = None

        if choice == "E":
            try:
                game = select_example_game()
            except (KeyboardInterrupt, Exception) as exc:
                console.print(f"[red]Error: {exc}[/]")
                continue

        elif choice == "C":
            try:
                game = enter_custom_game()
            except (KeyboardInterrupt, Exception) as exc:
                console.print(f"[red]Error: {exc}[/]")
                continue

        elif choice == "J":
            try:
                game = load_from_json()
            except (KeyboardInterrupt, Exception) as exc:
                console.print(f"[red]Error loading file: {exc}[/]")
                continue

        elif choice == "Q":
            console.print("\n[bold bright_cyan]Goodbye.[/]\n")
            sys.exit(0)

        if game is not None:
            equilibria = display_full_analysis(game)
            post_analysis_menu(game, equilibria)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold bright_cyan]Interrupted. Goodbye.[/]\n")
        sys.exit(0)