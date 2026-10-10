# Pilot 5 / 5b team transcripts: why teams did not beat best-of-N at equal spend (2026-10-11)

Reader: Opus subagent (full report was delivered as a message; scripts and data in the session scratchpad `transcripts-p5/`, regenerated timelines in analysis/runs/). One run per team cell; treat as observations.

**H1 diversity: supported in a narrower form.** Teams explored as many approaches as N independents, or more, in the first 600-900 s (dst: team of 2 built 4 solver classes, team of 4 built 5-6, against about 2.25 for a random pair of independents and about 2.5 for a random four). What they lost was diversity of finished candidates. Every team converged on one code lineage after one agent posted a fast number (team4-dst at 645 s: "agent_1 is way ahead"), while best-of-N keeps N separately engineered solvers. In 3 of 4 runs the converged lineage lacked a technique the better independents used, although a teammate had raised or built it: team2-dst declined its own working 7x7 stage (about 10% of transform time, about the size of its gap to the median independent); no geneig team built the serial bisection + Newton that scored best among independents (e3/e4).

**H2 coordination overhead: real, secondary.** Coordination calls plus reading teammates' folders took 9.6% (team2-dst), 19.2% (team4-dst), 7.6% (team2-geneig) and 23.3% (team4-geneig) of spend, against 0-0.5% for independents; teammate text in context cost 0.8-3.1%. Own-work calls per dollar matched independents. Teams reached their core solver later (team2-dst at $3.16 vs independents' median $1.97); among dst independents earlier arrival tracked score (Spearman -0.86, n=8). Eight agents in one 8-CPU box (team4-geneig) caused local-benchmark timeouts and one wrong diagnosis that spread by message.

**H3 idea flow: mixed.** team4-dst's selected solver combines code from 3 agents (that is the one team that matched best-of-4). team2-dst's is about 95% one agent's code; both geneig picks are one agent's kernel plus a within-noise change. Duplicated work in every team (three C kernels in team4-dst).

**Other.** team4-geneig's pick won selection on a lucky draw (4.14 at selection vs 3.17-3.40 elsewhere; five of six candidates near-copies). Little idling at the 80% gate (agents stopped 3-149 s after crossing it). Queue waits comparable to independents.

**Not supported / untested:** that coordination dollars alone explain the gap; that other selection or scoring would make teams win (only selected solvers were rescored).
