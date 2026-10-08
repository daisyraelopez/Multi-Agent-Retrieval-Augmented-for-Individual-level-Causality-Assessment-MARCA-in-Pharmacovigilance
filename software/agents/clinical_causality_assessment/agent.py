"""Agent 4 — clinical causality assessment report (CrewAI)."""
from __future__ import annotations

import re

from software import config
from software.agents.scientific_evidence_retrieval.agent import ensure_gemini_env

_TEMPLATE_VAR_RE = re.compile(r"{([^{}]+)}")

OUTPUT_STRUCTURE = """\
**Comprehensive Causality Assessment for <ADVERSE EVENT> Associated with <DRUG>**

**FINAL PLAUSIBILITY RATING:** HIGH / MEDIUM / LOW

**Detailed Causality Analysis:**
1. Temporal Plausibility
2. Biological Plausibility
3. Dechallenge/Rechallenge
4. Objective Evidence
5. Confounding Factors
6. History of Similar Reactions
7. Synthesis and Final Assessment"""


def build_causality_agent_and_task(drug: str, event: str, pair_context: str):
    """Return ``(agent, task)`` for the final causality report.

    In a sequential crew this task automatically receives the output of the
    preceding literature-summary task (Agent 1) as context.
    """
    from crewai import Agent, Task  # imported lazily: CrewAI is only needed when run_crew=True

    ensure_gemini_env()
    agent = Agent(
        role="Pharmacovigilance Causality Specialist",
        goal=f"Assess causality for '{event}' caused by '{drug}' using metadata and literature summary.",
        backstory="Experienced safety specialist.",
        llm=config.CREW_LLM,
        verbose=True,
    )
    task = Task(
        description=f"""
Use the following context to provide a comprehensive causality assessment for '{drug}' causing '{event}':

{pair_context}

Instructions for the agent:
- Follow the exact output structure:

{OUTPUT_STRUCTURE}

- Explicitly include Dechallenge/Rechallenge results, literature evidence (PMIDs), and patient metadata.
- Base all reasoning strictly on the provided information.

""",
        agent=agent,
        expected_output="Comprehensive causality assessment structured as requested",
    )
    return agent, task


def ensure_required_inputs(tasks: list, inputs: dict) -> dict:
    """CrewAI interpolates ``{name}`` placeholders found in task descriptions (abstracts
    may contain braces). Any placeholder without a value is filled with ``MISSING``."""
    required = set()
    for t in tasks:
        required |= set(_TEMPLATE_VAR_RE.findall(getattr(t, "description", "") or ""))
    for key in required:
        inputs.setdefault(key, "MISSING")
    return inputs


def task_text(task) -> str:
    out = getattr(task, "output", None)
    if out is None:
        return ""
    return out.raw if hasattr(out, "raw") else str(out)
