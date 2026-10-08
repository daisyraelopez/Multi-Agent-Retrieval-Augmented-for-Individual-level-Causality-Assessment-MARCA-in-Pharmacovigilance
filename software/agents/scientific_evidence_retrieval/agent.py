"""Agent 1 — scientific evidence retrieval and literature summary (CrewAI).

PubMed retrieval itself is deterministic (see ``pubmed.py``); the CrewAI agent
summarises the retrieved abstracts into a mechanism-of-action and clinical-evidence
summary that Agent 4 then uses.
"""
from __future__ import annotations

import os

from software import config


def ensure_gemini_env() -> None:
    """CrewAI's Gemini provider accepts GOOGLE_API_KEY or GEMINI_API_KEY."""
    if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]


def build_literature_agent_and_task(drug: str, event: str, literature_content: str):
    """Return ``(agent, task)`` for the literature summary step."""
    from crewai import Agent, Task  # imported lazily: CrewAI is only needed when run_crew=True

    ensure_gemini_env()
    agent = Agent(
        role="Expert Medical Searcher and Literature Summarizer",
        goal=f"Analyze PubMed results for '{drug}' and '{event}' to summarize MOA and clinical evidence.",
        backstory="Detailed researcher.",
        llm=config.CREW_LLM,
        verbose=True,
    )
    task = Task(
        description=f"Summarize PubMed content:\n{literature_content}",
        agent=agent,
        expected_output="MOA & Clinical Evidence Summary",
    )
    return agent, task
