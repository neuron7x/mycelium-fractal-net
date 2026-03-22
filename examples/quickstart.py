#!/usr/bin/env python3
"""MFN Quickstart — 10 lines, full pipeline."""

import mycelium_fractal_net as mfn

# Simulate
seq = mfn.simulate(mfn.SimulationSpec(grid_size=64, steps=32, seed=42))
print(repr(seq))

# Detect anomalies
print(repr(seq.detect()))

# Extract morphology features
print(repr(seq.extract()))

# Forecast 4 steps ahead
print(repr(seq.forecast(4)))

# Explain the decision
print(seq.explain().narrate())
