# fixed_engine_experiments_v1

These experiments are intended to test the corrected summary upper bounds in `IndexManagerSimple`.

## Recommended order

1. `config_F1_best_base.json`
   - `k=2000`, `itr=6`, `nb=300`, `nd=120`, `md=60000`, `heap_factor=0.12`
   - This is the best current configuration from the earlier experiments, now with upper-bound summaries.

2. `config_F2_softer_pruning.json`
   - `k=2000`, `itr=6`, `nb=300`, `nd=120`, `md=60000`, `heap_factor=0.10`
   - Test whether a truly safe upper bound combined with softer pruning reaches 0.9.

3. `config_F3_wider_index.json`
   - `k=2000`, `itr=6`, `nb=400`, `nd=150`, `md=60000`, `heap_factor=0.12`
   - Verify whether a slightly wider index improves recall after the summary fix.

4. `config_F4_tighter_pruning.json`
   - `k=2000`, `itr=6`, `nb=300`, `nd=120`, `md=60000`, `heap_factor=0.15`
   - Check whether the new upper-bound summaries allow tighter pruning without losing recall.

## Notes

- `heap_factor` should now be interpreted more safely because summaries are upper bounds.
- If `config_F1_best_base.json` reaches 0.9 or higher, then the pipeline fix is successful.
- If it does not, next focus should be on the index build strategy rather than the pruning threshold.
