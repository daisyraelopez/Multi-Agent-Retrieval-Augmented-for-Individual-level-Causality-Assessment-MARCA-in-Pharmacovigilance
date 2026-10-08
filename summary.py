"""Console summary of RAG results: expert score vs DeepSeek without and with RAG, per question."""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text
import re

_rc = Console()

# ── helpers ──────────────────────────────────────────────────────────────────
def _parse_score_reason(text: str):
    text = (text or "").strip()
    sm = re.search(r"Score\s*[:\-]\s*([^\n]+)", text, re.IGNORECASE)
    rm = re.search(r"Reasoning\s*[:\-]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    score  = sm.group(1).strip() if sm else "—"
    reason = rm.group(1).strip() if rm else text
    if len(reason) > 320:
        reason = reason[:317] + "…"
    return score, reason

_Q_LABELS = {
    "Q2":  "Q2 · Temporal Plausibility",
    "Q5":  "Q5 · Confounding Factors",
    "Q10": "Q10 · Sign vs Symptom",
}

# Colour only the score badge, never the row background
_SCORE_BADGE = {
    "2":  ("bold black on green",       "▲ +2"),
    "1":  ("bold black on chartreuse3", "▲ +1"),
    "0":  ("bold black on yellow",      "◆  0"),
    "-1": ("bold white on red3",        "▼ -1"),
    "-2": ("bold white on dark_red",    "▼ -2"),
}
def _score_badge(s: str) -> Text:
    key = s.strip().lstrip("+")
    style, label = _SCORE_BADGE.get(key, ("white", f"  {s}  "))
    return Text(label, style=style, justify="center")

# ── main print function ───────────────────────────────────────────────────────
def print_rag_summary(rag_results: list, max_ref_reason: int = 200):
    if not rag_results:
        _rc.print("[bold red]⚠  rag_results is empty — run the main agent cell first.[/]")
        return

    for entry in rag_results:
        drug = entry["drug"]
        pt   = entry["pt"]

        _rc.print()
        _rc.print(Panel(
            f"[bold cyan]Drug :[/bold cyan]  [bold white]{drug}[/bold white]\n"
            f"[bold cyan]AE   :[/bold cyan]  [bold white]{pt}[/bold white]",
            title="[bold yellow]▶  Drug / Adverse-Event Pair[/bold yellow]",
            border_style="yellow",
            expand=False,
        ))

        for qname, qlabel in _Q_LABELS.items():
            q = entry.get(qname, {})
            if not q:
                continue

            refs         = q.get("retrieved_refs", [])
            no_rag_txt   = q.get("version_no_rag",   "")
            with_rag_txt = q.get("version_with_rag", "")

            # ── expert score + reason from best KB ref ────────────────────
            expert_score  = "—"
            expert_reason = "(no matching KB case found)"
            if refs:
                best = refs[0]
                raw_score  = best.get(f"{qname}_h_score")
                raw_reason = best.get(f"{qname}_h_reason", "")
                expert_score  = str(raw_score) if raw_score is not None else "—"
                expert_reason = (raw_reason or "(no reasoning provided)")
                if len(expert_reason) > max_ref_reason:
                    expert_reason = expert_reason[:max_ref_reason - 1] + "…"

            # ── parse AI answers ──────────────────────────────────────────
            no_rag_score,   no_rag_reason   = _parse_score_reason(no_rag_txt)
            with_rag_score, with_rag_reason = _parse_score_reason(with_rag_txt)

            # ── comparison table  (NO row background colours) ─────────────
            tbl = Table(
                title=f"[bold magenta]{qlabel}[/bold magenta]",
                box=box.ROUNDED,
                show_lines=True,
                expand=True,
                header_style="bold white on dark_blue",
                row_styles=["", ""],   # alternate: plain + plain (no dark fill)
            )
            tbl.add_column("Source",              style="bold bright_white", width=24, no_wrap=True)
            tbl.add_column("Score",               justify="center",          width=10)
            tbl.add_column("Reasoning / Notes",   style="bright_white",      ratio=1)

            # ── row 1: expert human  (subtle left-border highlight via title colour) ──
            tbl.add_row(
                "[bold cyan]👤 Expert  (Human KB)[/bold cyan]",
                _score_badge(expert_score),
                f"[cyan]{expert_reason}[/cyan]",
            )

            # ── row 2: AI no RAG ──────────────────────────────────────────
            tbl.add_row(
                "[bold white]🤖 AI — no RAG[/bold white]",
                _score_badge(no_rag_score),
                f"[white]{no_rag_reason}[/white]",
            )

            # ── row 3: AI with RAG ────────────────────────────────────────
            tbl.add_row(
                "[bold bright_green]🔬 AI — with RAG[/bold bright_green]",
                _score_badge(with_rag_score),
                f"[bright_green]{with_rag_reason}[/bright_green]",
            )

            _rc.print(tbl)

            # ── reference details panel ───────────────────────────────────
            if refs:
                ref_lines = []
                for i, r in enumerate(refs, 1):
                    ref_lines.append(
                        f"  [dim][{i}][/dim] "
                        f"[cyan]{r['drug']}[/cyan] / "
                        f"[magenta]{r['pt']}[/magenta]  "
                        f"│ Expert {qname} = [bold yellow]{r.get(f'{qname}_h_score', '?')}[/bold yellow]  "
                        f"│ sim = [bold green]{r.get('sim', 0):.0%}[/bold green]  "
                        f"│ case_id = [dim]{r.get('case_id', '?')}[/dim]"
                    )
                _rc.print(Panel(
                    "\n".join(ref_lines),
                    title=f"[dim]KB references used for {qname}[/dim]",
                    border_style="dim",
                    expand=False,
                    padding=(0, 2),
                ))

        _rc.rule(style="dim")

    _rc.print(
        f"\n[bold green]✅  Summary complete — "
        f"{len(rag_results)} pair(s) printed.[/bold green]"
    )
