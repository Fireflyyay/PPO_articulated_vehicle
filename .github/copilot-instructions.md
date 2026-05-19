# Copilot Repository Instructions

You are working in a repository for autonomous navigation and parking of center-articulated vehicles.

The core architecture is:

- PPO selects discrete motion primitive IDs.
- The primitive library maps primitive IDs to executable motion primitives.
- The macro-action wrapper expands primitive IDs into low-level steering/speed sequences.
- The environment owns state update, articulated vehicle kinematics, collision checking, reward calculation, observation construction, success condition, and termination logic.
- Planner-side modules, fallback modules, Hybrid A*, sidecars, and takeover mechanisms must preserve this architecture unless explicitly instructed otherwise.

## Evidence rules

1. Do not invent repository files, functions, classes, configs, CLI flags, metrics, papers, citations, or experiment results.
2. Before claiming that a file/function/config exists, inspect or cite repository context.
3. If an interface is not found, write `"status": "not_found"` and explain what is missing.
4. Separate confirmed repository facts from assumptions and proposed changes.
5. Do not silently change PPO action-space semantics, primitive ID semantics, wrapper behavior, vehicle kinematics, reward scale, action mask semantics, or success condition.
6. Any algorithmic suggestion must specify input, output, failure mode, fallback, and validation method.
