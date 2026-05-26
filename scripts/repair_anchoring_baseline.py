"""Repair the 11 empty rows in gpt-oss-120b anchoring baseline CSV.

These rows came back empty because gpt-oss-120b hit max_tokens=1024 on long
chain-of-thought outputs. Re-call the actor at max_tokens=2048 and patch the CSV.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.reflexion.llm import make_client
from src.selfhelp.runner import SelfHelpRunner, _ADMIT_REJECT_SYSTEM, _parse_admit_reject

csv.field_size_limit(sys.maxsize)

CSV_PATH = "results/gpt-oss-120b_anchoring_baseline.csv"
SYSTEM = SelfHelpRunner._ANCHORING_BASE_SYSTEM + " " + _ADMIT_REJECT_SYSTEM
FIELDS = None  # capture from header

def main() -> None:
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames

    empties = [(i, r) for i, r in enumerate(rows) if not r["raw_answer"].strip()]
    print(f"Found {len(empties)} empty rows to repair.")

    client = make_client(provider="deepinfra", model="openai/gpt-oss-120b")

    repaired = 0
    still_empty = 0
    for i, row in empties:
        prompt_id = row["prompt_id"]
        sub_cond = row["sub_condition"]
        prompt = row["original_prompt"]
        print(f"\n[{repaired + still_empty + 1}/{len(empties)}] pid={prompt_id} sub={sub_cond} ...", end=" ", flush=True)

        resp = client.complete(
            SYSTEM, prompt,
            temperature=0.0, max_tokens=2048, json_mode=True,
        )
        if not resp.content.strip():
            still_empty += 1
            print(f"STILL EMPTY ({resp.usage.output_tokens} out tokens)")
            continue

        decision = "admit" if _parse_admit_reject(resp.content) == 1 else "reject"
        rows[i]["raw_answer"] = resp.content
        rows[i]["parsed_answer"] = decision
        repaired += 1
        print(f"OK  decision={decision}  out_tokens={resp.usage.output_tokens}  cost=${resp.usage.cost_usd:.5f}")

    print(f"\n--- summary ---  repaired={repaired}  still_empty={still_empty}")

    if repaired > 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote patched CSV to {CSV_PATH}")


if __name__ == "__main__":
    main()
