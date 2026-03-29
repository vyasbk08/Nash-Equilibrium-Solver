import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box
from typing import Dict, List, Optional, Set, Tuple

from game import NormalFormGame
from solvers import NashEquilibrium
from analysis import GameAnalyzer

console = Console()

THEME = {
    "nash":          "bold green",
    "br_p1":         "bold cyan",
    "br_p2":         "bold yellow",
    "pareto":        "bold magenta",
    "dominated":     "red dim",
    "header":        "bold white on dark_blue",
    "title":         "bold bright_white",
    "accent":        "bright_cyan",
    "pure_ne":       "bold green",
    "mixed_ne":      "bold blue",
    "section":       "bold bright_yellow",
    "value":         "bright_white",
    "positive":      "green",
    "negative":      "red",
    "neutral":       "white",
}


def _fmt(val: float, decimals: int = 4) -> str:
    if val == int(val):
        return str(int(val))
    return f"{val:.{decimals}f}"


def _payoff_text(
    p1_val: float,
    p2_val: float,
    is_nash: bool,
    br_p1: bool,
    br_p2: bool,
    is_pareto: bool,
) -> Text:
    t = Text()
    p1_str = _fmt(p1_val, 2)
    p2_str = _fmt(p2_val, 2)

    if is_nash:
        t.append("★ ", style="bold green")
        t.append(p1_str, style="bold green underline")
        t.append(", ", style="bold green")
        t.append(p2_str, style="bold green underline")
        t.append(" ★", style="bold green")
        return t

    if is_pareto:
        prefix, suffix = "◆ ", " ◆"
    else:
        prefix, suffix = "  ", "  "

    t.append(prefix, style="magenta")
    t.append(p1_str, style="bold cyan underline" if br_p1 else "white")
    t.append(", ", style="dim white")
    t.append(p2_str, style="bold yellow underline" if br_p2 else "white")
    t.append(suffix, style="magenta")
    return t


def display_payoff_matrix(
    game: NormalFormGame,
    highlight_nash: bool = True,
    highlight_pareto: bool = True,
):
    br1_mat, br2_mat = game.br_matrix()
    nash_mat = game.pure_nash_matrix()

    analyzer = GameAnalyzer(game)
    pareto_set = set(analyzer.pareto_optimal_outcomes()) if highlight_pareto else set()

    table = Table(
        title=f"\n[bold bright_white]{game.game_name}[/] — Payoff Matrix\n"
              f"[dim](★ = Nash Equilibrium  |  [cyan]P1 Best Response[/]  |  "
              f"[yellow]P2 Best Response[/]  |  [magenta]◆ Pareto Optimal ◆[/])[/]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold white",
        border_style="bright_blue",
        padding=(0, 1),
    )

    table.add_column(
        f"[bold]{game.player1_name} \\ {game.player2_name}[/]",
        style="bold white",
        justify="center",
        min_width=16,
    )
    for s2 in game.strategy_names_2:
        table.add_column(s2, justify="center", min_width=14)

    for i, s1 in enumerate(game.strategy_names_1):
        row = [Text(s1, style="bold white")]
        for j in range(game.n):
            p1_val, p2_val = game.payoffs_at(i, j)
            cell = _payoff_text(
                p1_val=p1_val,
                p2_val=p2_val,
                is_nash=bool(nash_mat[i, j]),
                br_p1=bool(br1_mat[i, j]),
                br_p2=bool(br2_mat[i, j]),
                is_pareto=(i, j) in pareto_set,
            )
            row.append(cell)
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()


def display_game_properties(game: NormalFormGame):
    props = []
    props.append(f"Shape: {game.m} × {game.n}")
    props.append("Zero-sum: " + ("[green]Yes[/]" if game.is_zero_sum() else "[red]No[/]"))
    props.append("Constant-sum: " + ("[green]Yes[/]" if game.is_constant_sum() else "[red]No[/]"))
    props.append("Symmetric: " + ("[green]Yes[/]" if game.is_symmetric() else "[red]No[/]"))

    content = "  ".join(props)
    console.print(Panel(content, title="[bold bright_yellow]Game Properties[/]", border_style="bright_yellow"))


def display_equilibria(equilibria: List[NashEquilibrium], game: NormalFormGame):
    console.print(Rule(f"[bold bright_cyan]Nash Equilibria ({len(equilibria)} found)[/]", style="bright_cyan"))

    if not equilibria:
        console.print(Panel("[red]No Nash Equilibria found.[/]", border_style="red"))
        return

    for k, ne in enumerate(equilibria, 1):
        ne_type = ne.equilibrium_type()
        color = "green" if ne.is_pure else "blue"

        lines = []

        p1_parts = []
        for i in range(game.m):
            if ne.strategy_p1[i] > 1e-10:
                pct = f"{ne.strategy_p1[i]:.4f}"
                p1_parts.append(f"[bold]{game.strategy_names_1[i]}[/]: [bright_white]{pct}[/]")
        p2_parts = []
        for j in range(game.n):
            if ne.strategy_p2[j] > 1e-10:
                pct = f"{ne.strategy_p2[j]:.4f}"
                p2_parts.append(f"[bold]{game.strategy_names_2[j]}[/]: [bright_white]{pct}[/]")

        lines.append(f"[cyan]{game.player1_name}:[/]  " + "  |  ".join(p1_parts))
        lines.append(f"[yellow]{game.player2_name}:[/]  " + "  |  ".join(p2_parts))
        lines.append("")

        sw = ne.social_welfare()
        sw_color = "green" if sw > 0 else "red" if sw < 0 else "white"
        lines.append(
            f"Payoff → [cyan]{game.player1_name}[/]: [{color}]{_fmt(ne.payoff_p1, 4)}[/]   "
            f"[yellow]{game.player2_name}[/]: [{color}]{_fmt(ne.payoff_p2, 4)}[/]   "
            f"Social Welfare: [{sw_color}]{_fmt(sw, 4)}[/]"
        )

        support_str = (
            f"Support: P1={{{', '.join(game.strategy_names_1[i] for i in ne.support_p1)}}}  "
            f"P2={{{', '.join(game.strategy_names_2[j] for j in ne.support_p2)}}}"
        )
        lines.append(f"[dim]{support_str}[/]")

        panel_content = "\n".join(lines)
        console.print(
            Panel(
                panel_content,
                title=f"[bold {color}]NE #{k} — {ne_type}[/]",
                border_style=color,
                padding=(0, 2),
            )
        )
    console.print()


def display_iesds(avail_1: List[int], avail_2: List[int], history: List[str], game: NormalFormGame):
    console.print(Rule("[bold bright_yellow]IESDS — Iterated Elimination of Strictly Dominated Strategies[/]", style="bright_yellow"))

    if not history:
        console.print(Panel("[green]No strictly dominated strategies found. IESDS yields the full game.[/]", border_style="green"))
        console.print()
        return

    elim_table = Table(box=box.SIMPLE_HEAVY, border_style="yellow", show_header=True, header_style="bold white")
    elim_table.add_column("Round", style="dim white", justify="center", min_width=7)
    elim_table.add_column("Elimination", style="white", min_width=55)

    for k, step in enumerate(history, 1):
        elim_table.add_row(str(k), step)

    console.print(elim_table)

    surviving_1 = [game.strategy_names_1[i] for i in avail_1]
    surviving_2 = [game.strategy_names_2[j] for j in avail_2]
    console.print(
        Panel(
            f"[cyan]{game.player1_name}:[/]  {', '.join(surviving_1)}\n"
            f"[yellow]{game.player2_name}:[/]  {', '.join(surviving_2)}",
            title="[bold green]Surviving Strategies After IESDS[/]",
            border_style="green",
        )
    )
    console.print()


def display_dominance(dom: Dict[str, List[int]], game: NormalFormGame):
    console.print(Rule("[bold bright_magenta]Dominance Analysis[/]", style="bright_magenta"))

    table = Table(box=box.SIMPLE_HEAVY, border_style="magenta", show_header=True, header_style="bold white")
    table.add_column("Player", min_width=12, style="bold white")
    table.add_column("Strictly Dominated", min_width=28, style="red")
    table.add_column("Weakly Dominated", min_width=28, style="yellow")

    sd1 = [game.strategy_names_1[i] for i in dom["strictly_dominated_p1"]]
    wd1 = [game.strategy_names_1[i] for i in dom["weakly_dominated_p1"]]
    sd2 = [game.strategy_names_2[j] for j in dom["strictly_dominated_p2"]]
    wd2 = [game.strategy_names_2[j] for j in dom["weakly_dominated_p2"]]

    table.add_row(game.player1_name, ", ".join(sd1) or "None", ", ".join(wd1) or "None")
    table.add_row(game.player2_name, ", ".join(sd2) or "None", ", ".join(wd2) or "None")

    console.print(table)
    console.print()


def display_pareto(game: NormalFormGame):
    console.print(Rule("[bold bright_magenta]Pareto Analysis[/]", style="bright_magenta"))
    analyzer = GameAnalyzer(game)
    pareto = analyzer.pareto_optimal_outcomes()
    sw_i, sw_j, sw_max = analyzer.social_welfare_maximizer()

    table = Table(box=box.SIMPLE_HEAVY, border_style="magenta", show_header=True, header_style="bold white")
    table.add_column(game.player1_name, style="bold cyan", min_width=16)
    table.add_column(game.player2_name, style="bold yellow", min_width=16)
    table.add_column("Payoffs (P1, P2)", min_width=18)
    table.add_column("Social Welfare", justify="right", min_width=14)
    table.add_column("SW Maximiser", justify="center", min_width=12)

    nash_mat = game.pure_nash_matrix()

    for i, j in pareto:
        p1v, p2v = game.payoffs_at(i, j)
        sw = p1v + p2v
        is_sw_max = (i == sw_i and j == sw_j)
        is_nash = bool(nash_mat[i, j])
        label = ""
        if is_sw_max:
            label = "[bold green]✔ MAX[/]"
        if is_nash:
            label += (" " if label else "") + "[green](Nash)[/]"
        table.add_row(
            game.strategy_names_1[i],
            game.strategy_names_2[j],
            f"({_fmt(p1v, 2)}, {_fmt(p2v, 2)})",
            f"{_fmt(sw, 2)}",
            label or "—",
        )

    console.print(table)
    console.print()


def display_minimax(game: NormalFormGame):
    if not game.is_zero_sum():
        return
    console.print(Rule("[bold bright_cyan]Minimax Theorem (Zero-Sum Game)[/]", style="bright_cyan"))
    analyzer = GameAnalyzer(game)
    v1 = analyzer.minimax_value_p1()
    v2 = analyzer.minimax_value_p2()
    console.print(
        Panel(
            f"[cyan]Maximin value (P1):[/]  [bold bright_white]{_fmt(v1, 4)}[/]\n"
            f"[yellow]Maximin value (P2):[/]  [bold bright_white]{_fmt(v2, 4)}[/]\n\n"
            f"[dim]By the Minimax Theorem (von Neumann 1928), these values coincide "
            f"in the mixed extension of every finite zero-sum game.[/]",
            title="[bold bright_cyan]Game Value[/]",
            border_style="bright_cyan",
        )
    )
    console.print()


def display_ne_vs_pareto(equilibria: List[NashEquilibrium], game: NormalFormGame):
    if not equilibria:
        return
    analyzer = GameAnalyzer(game)
    report = analyzer.nash_relative_to_pareto(equilibria)
    _, _, sw_max = analyzer.social_welfare_maximizer()

    console.print(Rule("[bold bright_green]Nash Equilibria vs Social Optimum[/]", style="bright_green"))

    table = Table(box=box.SIMPLE_HEAVY, border_style="green", show_header=True, header_style="bold white")
    table.add_column("NE #", justify="center", min_width=5)
    table.add_column("Type", min_width=18)
    table.add_column("P1 Payoff", justify="right", min_width=10)
    table.add_column("P2 Payoff", justify="right", min_width=10)
    table.add_column("Social Welfare", justify="right", min_width=14)
    table.add_column("SW Loss vs Max", justify="right", min_width=14)
    table.add_column("Pareto Optimal", justify="center", min_width=13)

    for k, item in enumerate(report, 1):
        ne = item["ne"]
        sw_loss = item["sw_loss_vs_max"]
        is_pareto = item["is_pareto_optimal"]
        pareto_str = (
            "[green]Yes[/]" if is_pareto is True
            else "[red]No[/]" if is_pareto is False
            else "[dim]—[/]"
        )
        loss_color = "green" if sw_loss < 0.001 else "yellow" if sw_loss < 2 else "red"
        table.add_row(
            str(k),
            ne.equilibrium_type(),
            _fmt(ne.payoff_p1, 3),
            _fmt(ne.payoff_p2, 3),
            _fmt(item["social_welfare"], 3),
            f"[{loss_color}]{_fmt(sw_loss, 3)}[/]",
            pareto_str,
        )

    console.print(table)
    console.print()


def display_full_analysis(game: NormalFormGame):
    console.print()
    console.print(Rule(f"[bold bright_white]{game.game_name}[/]", style="bright_white", align="center"))
    console.print()

    display_game_properties(game)
    display_payoff_matrix(game)

    from solvers import NashSolver
    solver = NashSolver(game)
    analyzer = GameAnalyzer(game)

    console.print(Rule("[bold bright_cyan]Computing Nash Equilibria via Support Enumeration...[/]", style="bright_cyan"))
    equilibria = solver.find_all_nash()
    display_equilibria(equilibria, game)

    display_minimax(game)
    display_ne_vs_pareto(equilibria, game)

    dom = analyzer.dominance_analysis()
    display_dominance(dom, game)

    avail_1, avail_2, history = analyzer.iesds()
    display_iesds(avail_1, avail_2, history, game)

    display_pareto(game)

    return equilibria
