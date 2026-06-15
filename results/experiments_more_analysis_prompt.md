# Prompt for next recall-improvement experiments

I ran five index/search configurations on the NQ dataset and got the following outcomes:

| Config | nb | nd | heap_factor | Recall@30 | Avg time/query (ms) | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| A | 500 | 150 | 0.60 | 0.4028 | 1.93 | Wider index breadth alone is not enough; recall stays very low and search is still too aggressive. |
| B | 1000 | 200 | 0.60 | 0.4172 | 2.20 | Even wider breadth helps only slightly; the pruning logic still suppresses most useful candidates. |
| C | 150 | 80 | 0.40 | 0.6211 | 4.12 | Softer pruning gives a clear recall jump with only a moderate time increase. |
| D | 150 | 80 | 0.20 | 0.7889 | 12.44 | This is the best recall so far, but the search is now slower. It suggests pruning is the biggest bottleneck. |
| E | 500 | 200 | 0.40 | 0.6542 | 6.08 | Combining wider index breadth with softer pruning improves recall, but not as much as expected. |

## What these results mean

1. The current pruning heuristic is the main limiter.
   - Config C and D show that reducing pruning aggressiveness strongly improves Recall@30.
   - This means many relevant candidates are being skipped before they are ever scored.

2. Broadening the index (larger nb / nd) helps, but not enough by itself.
   - Config A and B improve candidate coverage, yet recall remains below 0.45.
   - This suggests the index is still too narrow or too aggressively pruned to recover the true top-30.

3. The best trade-off so far is Config D.
   - It reaches 0.7889 Recall@30 at 12.44 ms/query.
   - That is a strong improvement over the other runs, but still below the minimum 0.9 target.

## What I would like you to recommend next

Please propose the next best experiment plan to reach at least 0.90 Recall@30 while keeping average query time as low as possible.

I want a practical recommendation that considers these constraints:

- Keep the current clustering setup as-is unless there is a clear reason to change it.
- Focus on the smallest number of new runs needed to find the best recall/time trade-off.
- Prioritize experiments that are likely to raise recall above 0.9 without exploding latency.

## Suggested directions to evaluate

1. Keep the pruning soft, but increase index breadth slightly.
   - Example direction: try nb in the 200–500 range and nd in 100–200 range while keeping heap_factor around 0.15–0.30.

2. Try a more targeted pruning schedule.
   - Instead of one fixed heap_factor for all queries, test whether a lighter threshold for difficult queries helps recall more than it hurts speed.

3. Measure whether the current index truncation is the real bottleneck.
   - If recall still stalls below 0.85, the index itself is too lossy and wider nb/nd is required.

4. Find the best recall/time knee.
   - The goal is not just maximum recall, but the smallest config that gets close to 0.9 with low average query latency.

## Final question

Based on these results, what is the next single best experiment (or short experiment set) you recommend to push Recall@30 from 0.79 toward at least 0.90 while keeping average query time under a reasonable threshold?
