"""Repair the 8 empty rows in gpt-oss-120b anchoring selfhelp CSV.

All 8 came from a single debiaser call (set_id=97) whose JSON response was
truncated at max_tokens=1024. With the bump to 2048 the call should now succeed.

Selfhelp anchoring rows in a session share the same `original_prompt` and all
come from ONE debiaser call — no inter-row cascade.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.reflexion.llm import make_client
from src.selfhelp.debiaser import SelfHelpDebiaser
from src.selfhelp.runner import _parse_json_response

csv.field_size_limit(sys.maxsize)

CSV_PATH = "results/gpt-oss-120b_anchoring_selfhelp.csv"
PROMPTS_PATH = "prompts/selfhelp_prompts.yaml"


def main() -> None:
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames

    # Find the affected session (all empties share the same original_prompt)
    affected_history = next(r["original_prompt"] for r in rows if not r["raw_answer"].strip())
    session = [(i, r) for i, r in enumerate(rows) if r["original_prompt"] == affected_history]
    n = len(session)
    print(f"Affected session: set_id={session[0][1]['prompt_id']}, n_students={n}")

    client = make_client(provider="deepinfra", model="openai/gpt-oss-120b")
    debiaser = SelfHelpDebiaser(client, PROMPTS_PATH)

    print(f"Calling debiaser (max_tokens=2048)...")
    revised_text, resp = debiaser.debias_decisions(affected_history, n_students=n)
    print(f"  output_tokens={resp.usage.output_tokens}  cost=${resp.usage.cost_usd:.5f}")

    # Parse the JSON array
    data = _parse_json_response(revised_text)
    decisions = data.get("decisions", []) if isinstance(data, dict) else []
    print(f"  parsed {len(decisions)} decisions; expected {n}")
    if len(decisions) < n:
        print(f"  WARNING: still short by {n - len(decisions)} students")
    while len(decisions) < n:
        decisions.append({})
    decisions = decisions[:n]

    repaired = 0
    for (idx, row), d in zip(session, decisions):
        if not isinstance(d, dict) or "admitted" not in d:
            print(f"  idx={idx} sub={row['sub_condition']}: still empty in new response")
            continue
        admitted = bool(d.get("admitted"))
        raw = json.dumps(d)
        rows[idx]["raw_answer"] = raw
        rows[idx]["parsed_answer"] = "admit" if admitted else "reject"
        rows[idx]["debiased_prompt"] = revised_text
        # First row in session carries the debiaser cost (matches original runner behavior)
        if repaired == 0:
            rows[idx]["cost_usd"] = f"{resp.usage.cost_usd:.8f}"
        else:
            rows[idx]["cost_usd"] = "0.00000000"
        repaired += 1
        print(f"  idx={idx} sub={row['sub_condition']}: parsed='{rows[idx]['parsed_answer']}'")

    print(f"\nRepaired {repaired}/{n}")
    if repaired > 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote patched CSV to {CSV_PATH}")


if __name__ == "__main__":
    main()
