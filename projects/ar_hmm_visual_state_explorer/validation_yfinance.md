# Validation notes

Suggested local checks after pulling this branch:

```bash
cd projects/ar_hmm_visual_state_explorer
python -m py_compile app.py arhmm_explorer.py
pytest -q
python run_walk_forward.py --ticker SPY --start 2010-01-01 --end 2020-01-01 --output data/outputs/spy_smoke --max-refits 2
```

The final command performs a minimal network-backed Yahoo Finance smoke run with only two walk-forward refits.
