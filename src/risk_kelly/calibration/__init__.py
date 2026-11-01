"""Parameter calibration engine — research sandbox (§3.2.4 / §4.6).

Derives Heimdall v1 OPEN parameters from historical data via walk-forward
calibration. Output becomes the signed config consumed by the live chassis.

Design constraints:
  - research/ sandbox: can import from contracts/ but NOT from core/
  - Pure functions only: same inputs → same outputs (reproducible, auditable)
  - Walk-forward mandatory: no in-sample fitting that the live engine hasn't seen
  - All outputs are fingerprinted: calibration result → signed config → core

Sub-modules:
  data_sources/  — historical time-series loaders (FRED, FINRA halt events)
  gate_calibration.py   — credit/vol/VIX threshold calibration (F-01/F-04)
  kelly_calibration.py  — fractional-Kelly coefficient (F-26)
  fill_probability_calibration.py — MWCB rung fill-prob (F-17)
  config_output.py      — fingerprinted calibration report (→ signed config)
"""
