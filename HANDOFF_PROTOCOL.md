# Handoff Protocol — Siamese Network Training Lab

This file governs *how* the implementation is executed. The technical spec lives in `siamese_infonce_implementation_plan.md` ("the plan"). The work is split into 7 phases, defined in `PHASE_BRIEFS.md`. **One phase per agent session. Hard stop at every gate.**

---

## For Ali (how to run the loop)

1. Start a **fresh agent session** (e.g., Claude Code in the repo root) for each phase. Fresh sessions keep context small; the repo + phase reports carry all state forward.
2. Give the agent four things every session:
   - the plan (`siamese_infonce_implementation_plan.md`)
   - the **poster image** (the original reference — the agent should cross-check formulas and panel content against it, not only the plan)
   - this protocol
   - the instruction: *"Execute Phase N from PHASE_BRIEFS.md. Stop at the gate."*
3. When the agent stops, read `results/phase_reports/phase_N.md`, verify the gate evidence yourself, and answer the **Learning checkpoint** questions in that phase's brief *before* approving. The checkpoints are the point of this project — each gate doubles as a self-quiz.
4. Approve → start the next session with Phase N+1. If a gate fails, stay on the phase until it passes; do not let the agent "fix it later."

## For the implementing agent (binding rules)

1. **Scope discipline:** implement only what the current phase brief lists under *Build*. Do not read ahead into later phases' code, do not scaffold future modules "while you're at it."
2. **Gate discipline:** run every gate check exactly as written, with the stated tolerances. Paste raw evidence (test output, printed numbers, figure paths) into `results/phase_reports/phase_N.md`. A gate is pass/fail — no "mostly passing."
3. **Stop discipline:** after writing the phase report, **stop and wait for human approval.** Do not begin the next phase in the same session even if asked ambiguously; ask for explicit confirmation.
4. **Honesty over tidiness:** if a result contradicts the plan's expected outcome, flag it prominently in the report. Contradictions are learning material here — never tune them away silently (plan §14.7).
5. The plan remains authoritative for anything a brief does not restate (formulas §6, defaults §7, layout §13, style rules §14).

## Phase map

| Phase | Title | Gate in one line |
|---|---|---|
| 0 | Environment & data | Tests green; identity-disjoint splits committed |
| 1 | Minimal pipeline | E0 overfit passes; epoch wall-clock measured |
| 2 | All losses & miners | Every unit-test vector passes exactly |
| 3 | Eval + viz suites | One baseline run yields a complete report + all figures |
| 4 | **DDP + global negatives** | Single-GPU vs 3-GPU loss **and gradients** match on a fixed batch |
| 5 | Run the experiment matrix | All E1–E8 results + cross-experiment comparison exist |
| 6 | Findings | Every poster claim answered with a figure and a number |

## Special callout: Phase 4

Phase 4's equivalence check is the single most important gate in the project. Cross-GPU negative gathering fails *silently* — training still runs, loss still decreases, but gradients from other ranks' anchors never reach the local embeddings, so you quietly get small-batch InfoNCE while believing you have large-batch InfoNCE (which would invalidate experiment E5 entirely). The Phase 4 brief specifies a two-stage test (isolated loss module on synthetic embeddings, then end-to-end) plus a list of the classic bugs to check for. Never skip or weaken this gate.
