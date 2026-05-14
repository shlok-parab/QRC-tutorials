# Quantum Reservoir Computing (QRC) Tutorials

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com?gitHubUrl=https://github.com/QuEraComputing/QRC-tutorials.git)

---

## Overview

![Overview of the quantum reservoir computing algorithm with QuEra's Aquila.](Images/QRC_overview.png)

The QRC pipeline consists of three stages:

1. **Classical preprocessing** – Convert raw data into a format that can be encoded on a neutral‑atom analog quantum computer (e.g., dimensionality reduction for images or feature engineering for time‑series).
2. **Quantum reservoir** – Encode the preprocessed features using one of three methods:
   - Global detuning pulse profile
   - Interaction‑strength modulation via atom‑position changes
   - Local detuning pulse pattern
   The quantum system evolves for a variable duration and is probed through projective measurements.
3. **Classical post‑processing** – Convert measurement outcomes into expectation values of local observables, forming the QRC embeddings. These embeddings are fed to a lightweight classical model (typically a linear SVM or linear regression) for training and inference.

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.9 – Install the required packages with:

  ```bash
  pip install -r requirements.txt
  ```

- **Bloqade SDK** – The Python package `bloqade` is installed automatically via the requirements above and provides the neutral‑atom simulation and hardware‑submission tools.

### Installation

```bash
# Clone the repository
git clone https://github.com/QuEraComputing/QRC-tutorials.git
cd QRC-tutorials/QRC-tutorials

# Set up a Python virtual environment (optional but recommended)
python -m venv env
# Windows:
env\\Scripts\\activate
# Unix/macOS:
source env/bin/activate
#or use conda
conda create --name qrc python=3.11
conda activate qrc

# Install dependencies
pip install -r requirements.txt
```

---

## Contents

| Notebook | Description |
| --- | --- |
| `QRC Demo MNIST.ipynb` | End‑to‑end QRC workflow on the MNIST digit dataset (Python). |
| `QRC Demo Timeseries.ipynb` | QRC applied to the Santa Fe laser time‑series prediction task (Python). |
| `QRC Demo Aquila Submission.ipynb` | How to submit QRC jobs to QuEra’s Aquila quantum processor using the Bloqade Python SDK. |

---

## Additional Resources

- **requirements.txt** – Lists all Python dependencies required for the notebooks.
- **Bloqade Documentation** – <https://bloqade.readthedocs.io/> for detailed API usage.

---

## Citation

If you use these tutorials in your work, please cite the original paper:

> QuEra Computing, *Large‑scale quantum reservoir learning with an analog quantum computer*, arXiv:2407.02553 (2024).

---

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.
