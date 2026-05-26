"""Repair the 2 downstream rows in gpt-oss-120b anchoring baseline whose history
was corrupted by the empty row at pid=949 sub=949:448082 (which defaulted to
'reject' when empty, then was repaired to 'admit').

Sequentially fixes idx+1 then idx+2, propagating any decision change.
"""
import csv
import re
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
EMPTY_KEY = ("949", "949:448082")


def replace_decision_for_student(prompt: str, profile: str, new_decision: str) -> str:
    """Replace 'Your decision: X' that immediately follows 'Student N: <profile>'."""
    pattern = r"(Student \d+: " + re.escape(profile) + r"\nYour decision: )(\w+)"
    return re.sub(pattern, r"\g<1>" + new_decision, prompt, count=1)


def main() -> None:
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames

    # Find the repaired empty row (highest cost among instances of EMPTY_KEY).
    instances = [(i, r) for i, r in enumerate(rows) if (r["prompt_id"], r["sub_condition"]) == EMPTY_KEY]
    instances.sort(key=lambda t: -float(t[1].get("cost_usd") or 0))
    repaired_idx = instances[0][0]
    repaired_decision = rows[repaired_idx]["parsed_answer"]  # 'admit'
    m = re.search(r"admit or reject the following student\?\n(.+)$", rows[repaired_idx]["original_prompt"], re.DOTALL)
    profile = m.group(1).strip()
    print(f"Repaired empty at idx={repaired_idx}, true decision='{repaired_decision}'")
    print(f"Profile: {profile[:80]}...")

    client = make_client(provider="deepinfra", model="openai/gpt-oss-120b")

    # --- Step 1: fix idx+1 (cascade row at position +1) ---
    i1 = repaired_idx + 1
    old_decision_i1 = rows[i1]["parsed_answer"]
    old_prompt_i1 = rows[i1]["original_prompt"]
    new_prompt_i1 = replace_decision_for_student(old_prompt_i1, profile, repaired_decision)
    assert new_prompt_i1 != old_prompt_i1, "Patch for idx+1 didn't change anything!"
    print(f"\n[Cascade 1/2] idx={i1} pid={rows[i1]['prompt_id']} sub={rows[i1]['sub_condition']}")
    print(f"  patched history: 'reject' → '{repaired_decision}' for the empty student")
    resp = client.complete(SYSTEM, new_prompt_i1, temperature=0.0, max_tokens=2048, json_mode=True)
    if not resp.content.strip():
        print("  STILL EMPTY — aborting"); sys.exit(1)
    new_decision_i1 = "admit" if _parse_admit_reject(resp.content) == 1 else "reject"
    rows[i1]["raw_answer"] = resp.content
    rows[i1]["parsed_answer"] = new_decision_i1
    rows[i1]["original_prompt"] = new_prompt_i1
    print(f"  old decision='{old_decision_i1}'  new decision='{new_decision_i1}'  "
          f"(changed={old_decision_i1 != new_decision_i1})  out_tokens={resp.usage.output_tokens}  cost=${resp.usage.cost_usd:.5f}")

    # --- Step 2: fix idx+2 (last in ordering) ---
    i2 = repaired_idx + 2
    profile_i1 = re.search(r"admit or reject the following student\?\n(.+)$", old_prompt_i1, re.DOTALL).group(1).strip()
    old_decision_i2 = rows[i2]["parsed_answer"]
    old_prompt_i2 = rows[i2]["original_prompt"]
    # Two patches: empty-student's decision (always), and idx+1's decision (if it changed)
    new_prompt_i2 = replace_decision_for_student(old_prompt_i2, profile, repaired_decision)
    if new_decision_i1 != old_decision_i1:
        new_prompt_i2 = replace_decision_for_student(new_prompt_i2, profile_i1, new_decision_i1)
        print(f"  also patched idx+1 student's decision in idx+2 history: '{old_decision_i1}' → '{new_decision_i1}'")
    assert new_prompt_i2 != old_prompt_i2, "Patch for idx+2 didn't change anything!"
    print(f"\n[Cascade 2/2] idx={i2} pid={rows[i2]['prompt_id']} sub={rows[i2]['sub_condition']}")
    resp = client.complete(SYSTEM, new_prompt_i2, temperature=0.0, max_tokens=2048, json_mode=True)
    if not resp.content.strip():
        print("  STILL EMPTY — aborting"); sys.exit(1)
    new_decision_i2 = "admit" if _parse_admit_reject(resp.content) == 1 else "reject"
    rows[i2]["raw_answer"] = resp.content
    rows[i2]["parsed_answer"] = new_decision_i2
    rows[i2]["original_prompt"] = new_prompt_i2
    print(f"  old decision='{old_decision_i2}'  new decision='{new_decision_i2}'  "
          f"(changed={old_decision_i2 != new_decision_i2})  out_tokens={resp.usage.output_tokens}  cost=${resp.usage.cost_usd:.5f}")

    # Write back
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote patched CSV to {CSV_PATH}")


if __name__ == "__main__":
    main()
