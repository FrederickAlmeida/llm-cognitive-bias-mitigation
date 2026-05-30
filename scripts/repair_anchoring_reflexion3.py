"""Repair empty rows in gpt-oss-120b anchoring reflexion step 3.

Empties hit finish_reason='length' with output_tokens=2048 — reasoning-overflow
(gpt-oss reasoning model exhausted its budget inside reasoning_content, leaving
content empty). Retry at max_tokens=4096; gpt-oss non-determinism usually yields
a shorter reasoning path that completes.

NO CASCADE. Unlike the baseline run, reflexion steps do NOT rebuild the
conversation history from the current step's decisions — each row's
`original_prompt` is the FROZEN baseline session history and is byte-identical
across baseline → step1 → step2 → step3. Verified empirically: 572/573
disambiguating rows carry the baseline decision in their history, not the
current step's. So an empty row's (re)answer never affects any other row;
each empty is repaired in isolation using its own (already-correct) prompt.

(The baseline repair scripts DO cascade, because baseline history is built live
during the sequential session. That distinction is the whole point.)
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(".env")

from src.reflexion.llm import make_client
from src.reflexion.memory import MemoryStore
from src.selfhelp.runner import SelfHelpRunner, _ADMIT_REJECT_SYSTEM, _parse_admit_reject

csv.field_size_limit(sys.maxsize)
CSV_PATH = "results/gpt-oss-120b_anchoring_reflexion_3.csv"
SYSTEM = SelfHelpRunner._ANCHORING_BASE_SYSTEM + " " + _ADMIT_REJECT_SYSTEM
FIELDS = ['bias_type','step','prompt_id','sub_condition','original_prompt',
          'prior_raw_answer','reflection_text','raw_answer','parsed_answer','cost_usd']


def memory_suffix(reflection_text: str) -> str:
    if not reflection_text.strip():
        return ""
    m = MemoryStore(); m.add(reflection_text)
    return "\n\nPrevious reflections on this decision:\n" + m.format_for_prompt()


def main() -> None:
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f); rows = list(reader); fields = reader.fieldnames

    client = make_client(provider="deepinfra", model="openai/gpt-oss-120b")
    empties = [i for i, r in enumerate(rows) if not r["raw_answer"].strip()]
    print(f"Found {len(empties)} empty rows: {empties}")

    for idx in empties:
        row = rows[idx]
        # Re-run with the row's OWN frozen original_prompt — no cascade, no patching.
        user = row["original_prompt"] + memory_suffix(row["reflection_text"])
        resp = client.complete(SYSTEM, user, temperature=0.0, max_tokens=4096, json_mode=True)
        if not resp.content.strip():
            print(f"[idx={idx}] STILL EMPTY at 4096 — needs manual handling")
            continue
        old_decision = row["parsed_answer"]
        new_decision = "admit" if _parse_admit_reject(resp.content) == 1 else "reject"
        rows[idx]["raw_answer"] = resp.content
        rows[idx]["parsed_answer"] = new_decision
        old_cost = float(row["cost_usd"] or 0)
        rows[idx]["cost_usd"] = f"{old_cost + resp.usage.cost_usd:.8f}"
        print(f"[idx={idx}] pid={row['prompt_id']} sub={row['sub_condition']}: "
              f"default '{old_decision}' → actual '{new_decision}'  out_tokens={resp.usage.output_tokens}")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    remaining = sum(1 for r in rows if not r["raw_answer"].strip())
    print(f"\nEmpties remaining: {remaining}")


if __name__ == "__main__":
    main()
