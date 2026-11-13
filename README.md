# Anonymous artifact snapshot — "When Is 1.58-bit Quantized Training Free?"

Artifact package for double-blind review. Author-identifying information has been removed.

## Layout

- `scripts/` — every experiment script referenced in the paper (pure NumPy, single-file,
  no framework). `extract_paper_seeds.py` regenerates all replication tables (Section 5)
  from the raw JSONs.
- `results/` — committed result JSONs with **complete evaluation trajectories**
  (500-step resolution), seeds, and configuration blocks. File naming:
  `<experiment>_results_<seedtag>.json`; `seed42` = original run, `s1/s2/s3` =
  independent seed replications.
- `PREREGISTRATION.md` — preregistered predictions verbatim, including the **falsified**
  one, with timestamps.

## Determinism

Same-seed relaunch produces byte-identical result files (pipeline is deterministic
end-to-end). Each JSON embeds its seed and config; hashes are recorded in
`results/hashes.txt`.

## Claim-evidence map

See the paper's Appendix A: every quantitative claim traces to one of the JSONs here.

## Environment

Python 3.8+, NumPy only. Each script is standalone (`python scripts/<name>.py`).
