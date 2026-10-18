# Opus results so far (v7, as of 2026-10-19)

Each group is one task: a diversified team of 4 Claude Opus 5.5 agents (effort xhigh) against 4 diversified independent Opus agents, $5 per agent. The independents' answer is the best of their candidates under the same two-stage selection rule the team gets. Scores are hidden-instance speedups over the task's reference solver (alone-reference primary score, median of 3 exclusive scorings of the pick, one scorer session per group).

| Task | Best of 4 independents | Team of 4 | Team / independents | Status |
|---|---|---|---|---|
| pagerank | 374.5 | 451.2 | 1.21x | Batch 1, final harness (v7b) |
| quantile_regression | 67.0 | 74.9 | 1.12x | Batch 1, final harness (v7b) |
| ode_lorenz96_nonchaotic | 106.5 | 184.8 | 1.74x | Calibration agents (16 Oct harness), re-selected and rescored on the final harness |
| stable_matching | 114.9 | 184.0 | 1.60x | Calibration; excluded: the team's box had the scorer's Zen 4 CPU, the independents' boxes Zen 1 |

- The two batch-1 groups ran entirely on the final harness: Zen 4 boxes for every agent, the scoring hardware stated in every prompt, selection after all agents, exclusive scoring in one session, the concrete-output guard.
- ode is a fair within-group comparison (all arms on the same t3a boxes), but its agents ran before the fixes, so it is reported separately. On re-selection the guard rejected the team's original pick (it returned an `array.array`, which upstream AlgoTune rejects) and 5 of its first 8 ranked candidates; a different valid team candidate (stage-1 rank 4) was picked. Both arms' scores rose under exclusive scoring (independents 96.3 to 106.5, team 152.5 to 184.8).
- Spread: each pick's 3 hidden scorings agree within 1-4%. The independents' best dev scores per agent ranged 1.4x (ode) to 2.6x (stable_matching) across the 4 agents. The team's best dev score exceeded every independent agent's on all four tasks.
- Uncertainty: with the 3 counted groups (log ratios 0.112, 0.186, 0.551), the geometric mean is 1.33x and the 95% t interval 0.74x to 2.38x; it includes no effect.
- Mixed team: one team of 2 Opus and 2 GPT-6.1 Sol agents was tried on ode in the calibration session, scoring 177.0 against that session's Opus team 152.5 and independents 96.3. Its Sol agents worked about two hours longer on the same dollar budget, and it was not rescored, so it is one exploratory run.
- Haiku 5.5 (v6, $0.10 per agent, 7 tasks x 2 repeats): team vs diversified independents 0.97x (6 of 14 groups), no clear team advantage. Model, budget and harness all differ from the Opus runs, so the two sets of results do not isolate model capability.

Records: `analysis/portfolio/20261018T213556-10608.jsonl` (batch 1), `20261018T181206-82548.jsonl` (ode re-selection), `20261016T191712-10624.jsonl` (stable_matching), `20261016T210428-31499.jsonl` (calibration ode incl. the mixed team). Reports: `analysis/v7-calibration.md`, `analysis/v7b-fixes.md`, `analysis/v6-main-results.md`.
