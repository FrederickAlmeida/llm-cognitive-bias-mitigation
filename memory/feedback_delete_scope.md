---
name: Scope deletions precisely
description: When deleting result files, target only the specific files that need to be deleted — never use broad wildcards that affect unrelated results
type: feedback
---

When deleting result files, always delete only the specific files that changed — not a broad wildcard. For example, if only anchoring feedback changed, delete only `*anchoring*reflexion*`, not `*reflexion*` which wipes all bias results.

**Why:** The user had valid framing/GA/primacy/status_quo reflexion results that took API calls to generate. A broad wildcard deleted all of them unnecessarily.

**How to apply:** Before running any `Remove-Item` or `rm` with a wildcard on results, list the matching files first and confirm the scope is correct.
