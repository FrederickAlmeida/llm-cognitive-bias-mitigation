#!/usr/bin/env python3
"""CLI entry point for running a single Reflexion trial."""
from __future__ import annotations

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv

# Allow running as `python scripts/run_reflexion.py` from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reflexion.agent import PromptLoader, ReflexionAgent
from src.reflexion.actor import make_actor
from src.reflexion.evaluator import make_evaluator
from src.reflexion.llm import make_client
from src.reflexion.memory import MemoryStore
from src.reflexion.reflection import SelfReflection
from src.reflexion.tools import ToolRegistry


def build_agent(config: dict) -> ReflexionAgent:
    prompts_path = os.path.join(config.get("prompts_dir", "prompts/"), "prompts.yaml")
    loader = PromptLoader(prompts_path)

    # Actor LLM
    actor_cfg = config["actor"]
    actor_llm = make_client(actor_cfg["llm_provider"], actor_cfg["llm_model"])

    # Tool registry (empty until PIX dataset tools are registered)
    tools = ToolRegistry()

    actor = make_actor(
        strategy=actor_cfg["strategy"],
        llm=actor_llm,
        prompt_loader=loader,
        tool_registry=tools,
        config=actor_cfg,
    )

    # Evaluator
    eval_cfg = config["evaluator"]
    if eval_cfg["type"] == "llm_judge":
        eval_llm = make_client(eval_cfg["llm_provider"], eval_cfg["llm_model"])
        evaluator = make_evaluator("llm_judge", llm=eval_llm, prompt_loader=loader)
    else:
        evaluator = make_evaluator("exact_match")

    # Self-reflection LLM
    refl_cfg = config["self_reflection"]
    refl_llm = make_client(refl_cfg["llm_provider"], refl_cfg["llm_model"])
    reflection = SelfReflection(refl_llm, loader)

    memory = MemoryStore(max_entries=config["memory"]["max_entries"])

    return ReflexionAgent(
        actor=actor,
        evaluator=evaluator,
        self_reflection=reflection,
        memory=memory,
        max_trials=config["reflexion"]["max_trials"],
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run a single Reflexion trial.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML.")
    parser.add_argument("--question", required=True, help="The question to answer.")
    parser.add_argument("--ground-truth", required=True, help="The expected correct answer.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    agent = build_agent(config)
    result = agent.run(args.question, args.ground_truth)

    print(result.summary())


if __name__ == "__main__":
    main()
