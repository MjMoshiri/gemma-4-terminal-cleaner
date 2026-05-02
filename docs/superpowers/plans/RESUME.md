# Resume Note — Gemma 4 E2B Terminal Cleaner

**Status when paused:** 2026-05-02. Brainstorming + plan-writing complete. Ready for implementation execution. User asked to pause for context compaction.

## Where we are

- ✅ Spec written, self-reviewed, committed: `docs/superpowers/specs/2026-05-02-gemma-terminal-cleaner-design.md`
- ✅ Plan written, self-reviewed, committed: `docs/superpowers/plans/2026-05-02-gemma-terminal-cleaner.md`
- ✅ Task persistence: `docs/superpowers/plans/2026-05-02-gemma-terminal-cleaner.md.tasks.json`
- ✅ 20 native tasks created (#9–#28), dependencies wired, all `pending`
- 🟡 **Awaiting user choice:** how to execute the plan (subagent-driven in this session vs parallel session)

## Decisions locked from brainstorming

| Topic | Decision |
|---|---|
| Cleanup style | **Strict lossless** (no truncation, no semantic summarization) |
| Model | **Gemma 4 E2B** (2.3B effective, 128K context, MLX 4-bit checkpoint `mlx-community/gemma-4-E2B-it-4bit`) |
| Hardware | MacBook Air **M4, 24 GB**, 10-core, fanless |
| Training method | **4-bit QLoRA via mlx-lm** (only realistic path on this hardware) |
| Data strategy | **Approach 1**: pure synthetic dirtifier-first, ~430 MB total. Small (500-pair) hand-curated real eval set. |
| Quality supervision | Claude (the agent) handles QA — spot-checks, eval curation, failure-mode review |
| Deployment scope | Pure text-in/text-out function. **No** shell hooks, **no** streaming, **no** multimodal in v1. |
| RTK reference | https://github.com/rtk-ai/rtk — rule-based Rust engine; we are improving over it via generalization, format robustness, and stricter losslessness |

## Where to pick up after compaction

1. Tell user: "Plan ready at `docs/superpowers/plans/2026-05-02-gemma-terminal-cleaner.md` with 20 tasks. How would you like to execute?"
2. Two options:
   - **Subagent-Driven (this session):** invoke `superpowers-extended-cc:subagent-driven-development` — dispatches a fresh subagent per task, reviews between tasks, fast iteration, stay in this session as coordinator.
   - **Parallel Session (separate):** open a new session in the worktree, invoke `superpowers-extended-cc:executing-plans` — batch execution with checkpoints.
3. **Recommended execution order** (already in plan §Self-Review and in `.tasks.json` `executionOrder`):
   `0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 11 → 10 → 12 → 8 → 9 → 13 → 14 → 15 → 16 → 17 → 18 → 19`
4. **Notable cross-task dependency:** Task 9 (real eval capture) imports `infer.ansi_strip` from Task 11 — that's why Task 11 comes before Task 9 in the order. Native task #18 has `blockedBy: [#20, #12]` to enforce this.
5. **The training task (Task 16)** is a 16–30-hour wall-clock run. It should be run overnight and tolerated for thermal throttling on the Air.

## Git state

- Branch: `main`
- Last commit: `dfc7345 Add implementation plan for Gemma 4 E2B terminal cleaner`
- Working tree: clean
- No remote configured (local-only project)

## Open implementation risks (already in spec §9 but worth keeping front-of-mind)

1. `mlx-lm` config-key drift between versions — Task 14 includes a `--help` verification step.
2. Synthetic→real distribution gap — guarded by 500-pair real eval set + iteration loop.
3. Lossless-guard atom regex precision — needs eyeball spot-check after Task 10 lands.
4. M4 Air thermal throttling on multi-day runs — `caffeinate` + frequent checkpointing baked in.

## Files committed since project start

```
docs/superpowers/specs/2026-05-02-gemma-terminal-cleaner-design.md
docs/superpowers/plans/2026-05-02-gemma-terminal-cleaner.md
docs/superpowers/plans/2026-05-02-gemma-terminal-cleaner.md.tasks.json
docs/superpowers/plans/RESUME.md  (this file — not yet committed)
```

(Project source code does not exist yet. Task 0 creates the skeleton.)
