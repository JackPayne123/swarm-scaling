# v6 main results (2026-10-12)

Plan `plans/v6-main.tsv`: 7 held-out AlgoTune tasks x 2 repeats x 3 arms. Haiku 5.5 at effort xhigh, $0.10 per agent, run to the 80% submit gate. Arms: 4 loose independents, 4 diversified independents (shared exploration text, one method priority each, no messaging), and a diversified team of 4 (same prompts plus the sharing line). Each arm is reduced to one solver by the same two-stage selection (`scripts/select_portfolio.py`). The score is the pick's hidden median of 3 scorings. All 126 rows exited 0. Measured model spend was $13.76 against a $16 cap.

| Group | Loose indep. | Diversified indep. | Diversified team |
|---|---|---|---|
| clustering_outliers r1 | 7.52 | 5.39 | 6.41 |
| clustering_outliers r2 | 2.69 | 4.54 | 6.58 |
| generalized_eigenvalues_complex r1 | 3.57 | 6.44 | 6.35 |
| generalized_eigenvalues_complex r2 | 4.77 | 3.95 | 6.43 |
| least_squares r1 | 1.00 (pick invalid on hidden) | 5.39 | 3.47 |
| least_squares r2 | 1.65 | 7.70 | 6.43 |
| pca r1 | 20.95 | 23.95 | 20.42 |
| pca r2 | 23.20 | 20.96 | 7.25 (shortlist best 12.75) |
| sinkhorn r1 | 6.89 | 7.67 | 7.34 |
| sinkhorn r2 | 7.46 | 8.17 | 9.72 |
| sparse_lowest_eigenvalues_posdef r1 | 4.51 | 3.70 | 6.16 |
| sparse_lowest_eigenvalues_posdef r2 | 6.12 | 8.13 | 7.58 |
| shortest_path_dijkstra r1 | 34.67 | 44.06 | 39.11 |
| shortest_path_dijkstra r2 | 40.60 | 38.25 | 38.74 |

Paired over the 14 groups (geometric mean of score ratios; the 95% interval is a t interval on log ratios, which treats the two repeats of a task as independent):

| Comparison | Wins | Ratio | 95% interval |
|---|---|---|---|
| Team vs diversified independents | 6/14 | 0.97 | 0.77-1.23 |
| Team vs loose independents | 10/14 | 1.33 | 0.92-1.90 |
| Diversified vs loose independents | 9/14 | 1.36 | 0.96-1.94 |

The team showed no gain over diversified independents at matched budget and selection. Both diversified arms beat loose independents in most groups, but at 14 groups neither interval excludes 1. The two loose least_squares groups (1.00 and 1.65) account for much of that gap.

To check before writing up:

- **least_squares loose r1:** the pick scored 40x on the 100 stage-2 instances and failed validity on hidden. Read why.
- **pca r2 team:** the pick scored 13.1 on stage 2 but 7.25 on hidden, while another shortlisted source scored 12.75 on hidden.
- **Lost records:** the generalized_eigenvalues_complex r2 and pca r2 portfolio records were overwritten when three selections wrote the same timestamped file. Their headline numbers come from the printed tables in `logs/_plans/reselect-v6-*.log`, and their job results are in `logs/_scorer/i-0d30a2ddff1fb1a5d/jobs.jsonl`. Output files now carry the pid.
- **Selection was interrupted:** the Mac lost network at 17:20, and the generalized_eigenvalues_complex r1/r2 and pca r2 selections died before any hidden job, so they were re-run on a fresh c7a.24xlarge scorer.
