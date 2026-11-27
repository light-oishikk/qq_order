# qq_order

Code for exact probability propagation, contextuality-by-default linear
programming checks, and ensemble simulations of sequential question-order
effects across quantum and classical belief-updating substrates.

## Setup

Python >= 3.9 is required. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the verification suite, generate simulation outputs, and compute summary
tables and figures:

```bash
python tests.py
python run_experiments.py
python analyze.py
```

Raw simulation outputs are written to `results/ledger/`. Summary statistics are
written to `results/summary.txt` and figures to `results/figures/`.

## Structure

- `quantum_core.py`: Density-matrix operators, unsharp measurement channels, and
  single-qubit projective formulas.
- `substrates.py`: Two- and three-question quantum, classical, and reservoir
  models and readout-retuning routines.
- `analysis.py`: Global-section and Contextuality-by-Default linear programs and
  JSONL ledger utilities.
- `tests.py`: Analytical and numerical consistency verification checks.
- `run_experiments.py`: Batch driver for all experimental ensembles and
  adversarial optimization runs.
- `analyze.py`: Summary statistics, analytical checks, and figure generation.
