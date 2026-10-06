# Task families

Assessed 2026-10-05 from current system cards (Opus 5.5, Sonnet 5.5, GPT-6 Astra, GPT-6.1 Sol, GPT-5.6 Sol, Gemini 4 Argon), Artificial Analysis index components, and repo inspection. Criteria: verifiable automatically, continuous or partial-credit score, room for many approaches, public, affordable many times with a cheap model.

## Chosen

### AlgoTune (primary)

- NeurIPS 2025, arXiv 2507.15887, repo github.com/oripress/AlgoTune (MIT), site algotune.io. 154 CPU tasks; write a faster solver matching the reference output (SciPy, sklearn, CVXPY). Score = speedup; official aggregate = harmonic mean, invalid or slower = 1x.
- Harbor/Inspect: `inspect_harbor/algotune`. Harbor adapter uses interleaved timing, 100 instances x 10 reps, 8 CPUs / 16 GB.
- Leaderboard stopped March 2026 (last result commits 11-12 Mar: GPT-5.4, "AlgoTune Lite" subset added). Top: GPT-5.2 2.05x, Gemini 3.1 Pro 2.02x, GPT-5.4 1.85x, Opus 4.5 1.77x. No 2026-frontier entries; no card reports it. Repo pushed last 2026-06-24 (README).
- Cheap-model proxies: gpt-5-mini 1.38x, gpt-oss-120b 1.41x, qwen3-coder 1.44x, GLM-4.5 1.52x (16-33 of 154 tasks failing).
- Original AlgoTuner budget: $1 per task (SpendTracker), 5-message history.
- Headroom: 32 tasks with best-model speedup 1.5x-10x and median at least 1.05x (e.g. chebyshev_center, randomized_svd, pca, shortest_path_dijkstra, set_cover_conflicts, btsp, min_dominating_set, qp, job_shop_scheduling, sinkhorn, least_squares, tsp, vehicle_routing, lasso, clustering_outliers). 20 tasks never exceed 1.1x for any model; drop them. Speedups are heavy-tailed (max 3,084x on ode_seirs), so use per-task log speedup.
- Validators have been gamed (commits tightening PCA "rotated subspace shortcuts", MVEE shape checks, firls validation). Audit winning solutions.
- Contamination: released July 2025, likely in 2026 training data. Affects arms roughly equally. The authors' "surface-level optimisations, no algorithmic innovation" finding may not hold for 2026 models.
- Day-1 check: can agents validate correctness mid-run in the Harbor version (evaluator lives in /tests, mounted only at scoring)?

### Harvey LAB data rooms (second family)

- github.com/harveyai/harvey-labs (MIT, v1.2.0, pushed 2026-10-04). Repo: 2,010 tasks, 27 areas, 114,437 atomic binary criteria (median 54 per task). Harbor dataset `harveyai/lab`: 1,251 tasks, the v1.0 launch snapshot (2026-05-08), with no `diligence/*`, `firm-knowledge/*` or `contracts/*` tasks (list in `task-data/harvey_harbor_tasks.json`).
- Use only the 11 `diligence/*` data rooms: 2,600-4,060 files (120-318 MB, tens of millions of tokens), 438-1,114 criteria each; deliverable is one red-flags report whose criteria are findings in specific documents. Recall problem, decomposes by folder, cannot fit in any context. No published results on them. Per-task stats: `task-data/lab_rows.json`.
- Standard tasks are unusable: all-pass 3-14% for most models (±4 points on 120 tasks), criterion pass 88-95% even for cheap models (AA: GLM-5.2 91.0%, Sonnet 5.5 93.1%).
- No mid-run verification: the rubric (`task.json`) is never mounted; the `finish` tool only checks deliverable files exist.
- Scores for reference: Vals held-out all-pass, GLM 5.3 Flash 6.67% at $0.57/test (52 min); Sonnet 5.5 2.92% at $16.61/test. AA LAB-AA (120 private tasks, Gemini 3.1 Pro judge, Stirrup harness): GLM-5.2 7.5% / 91.0% at $1.30/task.
- Risks: judge cost at data-room scale; no compaction in the repo harness; public rubrics (contamination); the independent-arm combination rule (PLAN.md open question).
- Also available: 250 `firm-knowledge` tasks sharing one 9,288-file store (median 6 criteria). Unassessed.

## Phase 2

### ProgramBench

- github.com/facebookresearch/ProgramBench (MIT, Meta), arXiv 2605.03546, programbench.com, per-test registry github.com/ProgramBench/submissions. 200 public tasks (107 Rust, 46 Go, 33 C, 12 C++), 247,444 active hidden tests. Score = mean fraction of tests passed. No public 166-task "golden" list (cards use 166).
- `inspect_harbor/bencalvert04_programbench` exists.
- Registry under mini-swe-agent (mean pass / cost per task): Haiku 4.5 0.30 / $0.80, Gemini 3 Flash 0.33 / $0.29, Sonnet 4.6 0.49 / $26.6, Gemini 3.7 Flash 0.62 / $2.05, GLM-5.2 0.65 / $25.5, Opus 5 xhigh 0.75 / $51. Model calls per task 80-440. Cross-model Spearman of per-task scores 0.5-0.77.
- Subsets: 52-task `task-data/cand.json`; 20-task `task-data/core_subset.json` (bat x2, oranda, scc, rhit, cheat, sox, datasurgeon, xh, rust-sloth, git-trim, go-critic, goimports-reviser, svgbob, lazygit, seqtk, caps-log, marmite, monolith, zk). Skip ffmpeg, gromacs, lnav, php.
- Needs Linux x86 Docker; images ~1 GB each. Guard against wrapping the reference binary (eval removes known hashes; unverified).
- Anthropic matched-token baseline: single 20M vs 5 x 4M (Opus 5.5 card).

## Considered and rejected

| Eval | Source | Why not |
|---|---|---|
| BrowseComp, DRACO, WANDR | OpenAI / Anthropic cards | Search-style; Jack finds uninteresting; Ord already has BrowseComp λ |
| Terminal-Bench 4.0 / Science | Cards, AA | 66-70 pass/fail tasks; OpenAI already has TB swarm data |
| FrontierSWE v2 | Anthropic, Google | ~20 h per task |
| PostTrainBench, NanoGPT speedrun | Google, OpenAI | 5-10 H100-hours per run |
| MLE-Bench | OpenAI | Kaggle-scale runs |
| OSWorld 2.x | Cards | GUI, slow, costly |
| ITBench-AA | AA | Kubernetes incident environments, setup too heavy for a week |
| GDPval-AA | AA | Pairwise Elo judge panel, noisy and costly |
| SciCode | AA | Score barely moves with capability (GLM-5.3-Flash 0.516, Luna 0.546, Sol 0.542, Sonnet 5.5 0.57-0.61); no mid-run verifier; sequential subproblems; near single-turn |
| GDP.pdf, GraphWalks | AA, Google | Single-turn / synthetic, little to coordinate |
| CritPt | AA | Answers private; AA grading server allows 10 submissions/day, full 70-problem batch only, aggregate scores only; scores mostly grader error (arXiv 2609.13009) |
| Lean milestones | Anthropic (internal) | No public weighted-milestone benchmark; building one ~1 week (RLMEval as raw material); ArXivLean: open models solve 0-1 of 41 |
| Cyber (ExploitBench, ExploitGym, SEC-Bench Pro) | Cards | Containment, cost, publication sensitivity |
| Bio (Dyno RNA/AAV) | Anthropic | Hazard |
| Private evals (AA-Briefcase, AutomationBench held-out, CoBench, KernelGen, Internal Research Debugging) | Various | Not available |

## Alternative if a second verifiable family is wanted

- Frontier-CS (arXiv 2512.15699, github.com/FrontierCS/Frontier-CS): 172 open-ended algorithmic problems with partial scoring, plus 66 research problems. Available as `inspect_evals/frontier_cs` and `inspect_harbor/yanagiorigami_frontier_cs`. Recommended by the Sol review; Park et al. used a Frontier-CS polyomino problem. Unassessed for cheap-model headroom.
