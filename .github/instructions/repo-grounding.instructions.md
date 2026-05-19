---
applyTo: "src/**/*.py"
---

# Repository-grounded coding rules

When discussing or modifying Python code in this repository:

1. Do not invent files, functions, classes, attributes, config names, command-line flags, or log metrics.
2. If a requested integration point is not visible in the current context, report it as unknown instead of guessing.
3. Any proposed code change must identify:
   - target file;
   - target function/class;
   - input;
   - output;
   - caller;
   - callee;
   - fallback behavior;
   - validation command.
4. Preserve the PPO + motion primitive + macro-action wrapper architecture.
5. Preserve discrete action-space semantics unless explicitly asked to redesign them.
6. Preserve center-articulated vehicle kinematics and articulation-angle consistency.
7. Any Hybrid A*, planner guidance, takeover, fallback, or sidecar mechanism must state whether its generated transitions enter the PPO buffer.
8. Any training-related change must state how it affects:
   - reward;
   - observation;
   - action mask;
   - curriculum;
   - termination;
   - success condition;
   - logging.
9. Run codes under the conde environment named "HOPE".