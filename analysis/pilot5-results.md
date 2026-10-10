# Pilot 5 and 5b results (2026-10-10)

Final scores: scoring-fixes scorer (ef90ec4: glibc trimming off, no CPU quota, cpu_dma_latency 0, no cached baseline), c7a.12xlarge, 4 slots, every selected solver scored 3 times, median reported. Source: analysis/rescore/20261010T173622.jsonl. All 57 scorings valid; reference_inflation 0.965-1.051; per-solver spread 0.8-6.5%.

Setup: Claude Opus 5.5, effort xhigh, run to budget (submit refused below 80% of the agent's budget), 1 h prompt cache, 4 CPUs per agent, remote scorer. One run per team cell. Best-of-N = mean over all N-subsets of independents of the best member.

## dst_type_II_scipy_fftpack, $5 per agent (pilot 5)

| arm | total budget | median score |
|---|---|---|
| independent x8 | $5 each | median 27.9 (25.1-33.6) |
| best of 2 independents | $10 | 29.9 |
| team of 2 | $10 | 26.3 |
| one agent, $10 | $10 | 29.8 |
| best of 4 independents | $20 | 31.4 |
| team of 4 | $20 | 32.3 |

## generalized_eigenvalues_real, $2 per agent (pilot 5b)

| arm | total budget | median score |
|---|---|---|
| independent x6 | $2 each | median 2.94 (2.18-3.32) |
| best of 2 independents | $4 | 3.13 |
| team of 2 | $4 | 2.64 |
| best of 4 independents | $8 | 3.30 |
| team of 4 | $8 | 2.77 |

## Reading

At equal spend, teams did not beat picking the best of the same number of independent agents: the team of 4 roughly matched best-of-4 on dst (+3%) and fell below the median independent on the eigenvalue task; the team of 2 was below best-of-2 on both. On dst, one agent with $10 matched best-of-2 at $5 each. Pilots 3 and 4, where agents could stop early, showed large team advantages (team of 4 beat every independent by about 20%); with spend equalised that advantage did not appear. One team run per cell and two tasks: a pointer, not a result.
