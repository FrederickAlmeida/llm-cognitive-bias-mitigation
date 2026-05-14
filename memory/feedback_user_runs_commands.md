---
name: User prefers to run commands themselves
description: Don't run long/background commands on behalf of the user — give them the command to run
type: feedback
---

Don't run commands on behalf of the user when they are long-running or background tasks. Just provide the command and let them run it.

**Why:** User prefers to control when and how long-running scripts are executed.

**How to apply:** When asked to run a script (especially evaluation pipelines), output the command instead of executing it.
