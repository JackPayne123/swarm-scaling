# Replication of Ord's swarm-scaling λ

## Method

Data is `research/sources/openai/multiagent_charts.json`, GPT-5.6 Sol series only, budget = total output tokens across all agents. BrowseComp and SEC-Bench Pro have 1, 4 and 16 agents at five efforts (15 points each). Terminal-Bench 2.1 has 1 and 4 agents only (10 points) plus four GPT-5.5 points that are excluded. Treating those four as solo points moves its linear λ from 0.48 to 0.59.

N agents match one agent given N^λ times the per-agent budget b. With a solo curve score = a + s ln b, N agents with total tokens T = Nb score a + s ln T + s(λ-1) ln N. Regressing score on ln T and ln N with coefficients s and c gives λ = 1 + c/s. Equivalently, at equal score the N-agent curve sits (1-λ) ln N to the right in log total tokens. We fit this pooled model and Ord's same-slope version (one intercept per N, one λ per pair of N from the horizontal offset), on linear and logit score. A synthetic check from the definition recovered the true λ for every estimator (scratch, not kept).

## Results

Pooled model, 90% interval (OLS delta method).

| Benchmark | Ours, linear | Ours, logit | Ord | Sol check (linear / logit) |
|---|---|---|---|---|
| BrowseComp | 0.650 [0.45, 0.85] | 0.676 [0.56, 0.80] | 0.68 [0.63, 0.76] | 0.650 / 0.676 |
| SEC-Bench Pro | 0.565 [0.52, 0.61] | 0.572 [0.53, 0.62] | 0.57 [0.52, 0.61] | 0.565 / 0.572 |
| Terminal-Bench 2.1 | 0.481 [0.39, 0.58] | 0.494 [0.41, 0.58] | 0.48 [0.40, 0.57] | 0.481 / 0.494 |

Point estimates equal the Sol check to three decimals and Ord's to within 0.03. A residual bootstrap agrees with the delta method to 0.03. Our SEC-Bench Pro and Terminal-Bench intervals match Ord's to 0.01. His BrowseComp interval (width 0.13) is narrower than ours (0.24 to 0.40).

Matched tokens, with the solo curve interpolated at each swarm point inside the solo token range. Of 8 such points (BrowseComp 2, SEC-Bench Pro 5, Terminal-Bench 1) the single agent leads at all 8, by 0.22-3.85, 8.76-23.33 and 6.25 points. This is the λ < 1 offset read at fixed tokens, not a second confirmation.

## What changes and what does not

SEC-Bench Pro fits well (R² 0.99). Slopes relative to solo are 1.04 and 0.89, pooled λ is 0.54-0.58 across effort subsets, and 1 to 4 and 4 to 16 agents give 0.547 and 0.584.

BrowseComp is not well identified. The solo curve saturates (linear R² 0.84, residual SD 2.7 points; logit 0.94), and swarm slopes are 0.55 and 0.41 of the solo slope, so the equal-slope premise fails. Step λ is 0.82 for 1 to 4 agents and 0.47 for 4 to 16. Dropping low effort raises pooled λ to 0.81, dropping max gives 0.63. Separate-slope readings span 0.44 to 1.01, so a single 0.68 hides a wide range.

Terminal-Bench tests only 1 to 4 agents, so Ord's claim that 1 to 4 and 4 to 16 agree cannot be checked on it. Pooled λ is 0.48-0.52 dropping max or low effort and 0.585 [0.34, 0.83] with both dropped (6 points).

The budget unit matters. With API cost in place of output tokens, linear λ rises to 0.71, 0.65 and 0.54, because cost per 1k output tokens varies with effort (SEC-Bench Pro solo, $0.095 to $0.202).

## Caveats

The points are aggregates and the file has no task counts, repeats or per-task scores. Every interval is a fit interval from 10-15 points. It treats residuals as independent although points lie on smooth curves, and it ignores sampling error inside each point. It is not a task-resampled interval and understates uncertainty.

The `latency_s` field is in minutes. Every label reads "minutes" and matches the value, and implied solo output rates are 30-62 tokens/s read as minutes against 1,800-3,700 read as seconds. Ord's speed claim (4 agents in half the time at matched score) is not tested here.
