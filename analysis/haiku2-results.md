# Haiku 2 results (2026-10-11)

Claude Haiku 5.5, effort xhigh, $0.25 per agent, run to budget (submit refused below 80% spend), loose prompt, 4 CPUs per agent, remote scorer (fixed timing setup). Per task: 4 teams of 4, 4 teams of 2, 10 independents. Scores are medians of 3 rescorings (dst: analysis/rescore/20261011T154928.jsonl; generalized eigenvalues: analysis/rescore/20261011T182535.jsonl); all valid. Best-of-N = dev-selected (the independent with the best recorded dev score in each N-subset), averaged over all N-subsets of the 10 independents. optimal_advertising was cancelled (credit moved to v6). Measured model spend $14.12.

| task | independents (median, range) | best-of-2 | team of 2 (mean; runs) | best-of-4 | team of 4 (mean; runs) |
|---|---|---|---|---|---|
| dst_type_II_scipy_fftpack | 7.2 (6.5-12.3) | 9.50 | 9.94 (7.0, 8.3, 12.1, 12.3) | 11.01 | 8.68 (4.1, 7.2, 11.6, 11.8) |
| generalized_eigenvalues_real | 1.94 (1.17-2.56) | 2.21 | 2.03 (1.4, 1.8, 2.1, 2.8) | 2.42 | 1.90 (1.5, 1.5, 2.3, 2.4) |

Team of 2 matched best-of-2 on dst (+5%) and fell 8% short on the eigenvalue task; team of 4 fell 21% short on both. Team runs spread as widely as single agents (dst team of 4: 4.1-11.8), so four repeats per cell cannot resolve gaps of this size. Transcripts for dst: analysis/haiku2-dst-transcripts.md.
