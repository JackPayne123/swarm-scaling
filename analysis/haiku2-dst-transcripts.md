# Haiku 2, dst: transcript findings (2026-10-11)

Reader: Opus subagent with four forked readers; all 24 team-agent timelines read in full, quotes spot-checked. Scratch data in the session scratchpad `transcripts-h2dst/`. Scores are rescored medians (analysis/rescore/20261011T154928.jsonl).

**One technique decides the score: float32.** Every selected solver that casts to float32 and runs two threaded `scipy.fft.dst` passes (returning float32) scores 11.1-12.3x; every float64 solver scores 4.7-8.3x. No final solver used C, numba or another algorithm. All 34 agents considered float32 and most measured it (error 1.9-2.5e-7 against the 1e-6 tolerance, 1.5-2.2x faster). Those who declined cited risk or fidelity ("the reference returns float64", "feels like gaming the validity check"). Independents behaved the same: 3 of 10 adopted.

**Teams are a best-of-N over what their agents publish.** Selection takes the fastest valid published candidate, so a team scores about 12x if any one agent publishes float32. Teams of 2 had a float32 publisher in 2 of 4 runs (a random pair of independents: 53%); teams of 4 in 2 of 4 (a random four: 83%). Under independence with p = 0.3 per agent, two of four team-of-4 runs lacking an adopter has probability about 0.25: the shortfall is consistent with chance at n = 4. Weak signs of suppression (team4-r3 had three "NOT using float32" broadcasts; a fidelity message turned one recommendation) and of spread (in three of four float32 teams a second agent adopted it by message, and twice the adopter's version was selected). In all three 7x team runs float32 was raised and dropped, not missed. Four runs carried wrong negative findings by message.

**team4-r4 (4.05) was a selection accident.** The rule picked a candidate on a 0.7% dev margin whose solver calibrates its thread count once on its first call (8 vs 4 workers) and keeps the choice; at final scoring it chose 4 workers in 3 of 4 scorings (4.0x) while dev chose 8 in 18 of 18 (8.3-8.6x). The same chain without calibration scores about 7.1x. Why the final verifier favours 4 is not established.

**Coordination is cheap for Haiku:** 1.3-3.0% of team spend (Opus pilot 5: 7.6-23.3%). Messages are one-shot findings reports; task list unused. Teams did not lose class diversity overall (2.5 classes per team of 4 vs 2.2 for a random four).

**Padding to the 80% submit gate dominates Haiku spend.** Agents finished real work by about $0.02-0.08 of their $0.25 and spent the rest on refused submits, check_budget and wait loops, repeated dev_evals and text-only turns (one agent spent $0.21 on text-only turns refusing to pad). The extra budget bought padding, not exploration, so for Haiku "equal spend" did not mean equal search.

**Dev-to-final drift:** dev overstates float64 solvers (final/dev 0.80-0.85) but not float32 (0.93-1.00), and an allocator tweak (mallopt) helps float64 only at final scoring, so dev rankings among float64 candidates are close to arbitrary.
