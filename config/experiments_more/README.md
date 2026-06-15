# Additional recall experiments

These configs are meant to be run after the current baseline on the same dataset.

Use them in this order:
1. config_exp_A_wider_500_150.json
2. config_exp_B_wider_1000_200.json
3. config_exp_C_soft_0_40.json
4. config_exp_D_soft_0_20.json
5. config_exp_E_wider_soft.json

What each one tests:
- A: wider index breadth, same pruning
- B: even wider index breadth, same pruning
- C: softer pruning, same index width
- D: very soft pruning, same index width
- E: combined wider index + softer pruning

Expected effect:
- Wider nb/nd should increase recall by keeping more candidate blocks/documents.
- Softer heap_factor should reduce false pruning and help recall.
- The combination E is the most likely to move recall toward 0.9, but may increase latency.
