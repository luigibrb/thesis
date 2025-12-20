# Assessing Concept Drift in Malware Classifiers via Conformal Prediction

This repository contains the source code and experiments for the Master's degree thesis by Luigi Barba, titled "Assessing Concept Drift in Malware Classifiers via Conformal Prediction".

## Table of Contents
- [Abstract](#abstract)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Data](#data)
- [Usage](#usage)
- [Running Tests](#running-tests)

## Abstract
TBD

## Repository Structure

The repository is organized into two main packages within the `src/` directory:

- **`src/experiment/`**: Contains the primary code for the experiments conducted in the thesis. This includes data loading (`DataLoader.py`), model definition (`Model.py`), training and evaluation logic (`Trainer.py`), and the main experimental pipeline (`main.py`).

- **`src/graphs/`**: Contains Python scripts dedicated to generating the graphs and figures used in the thesis document.

```
/
├── data/
│   └── .gitignore
├── src/
│   ├── experiment/
│   └── graphs/
├── tests/
├── .python-version
├── pyproject.toml
└── uv.lock
```

## Setup and Installation

This project uses `uv` for package management and requires **Python 3.12 or higher**.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd thesis
    ```

2.  **Create a virtual environment:**
    It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    # Create the environment
    python -m venv .venv

    # Activate it (on Linux/macOS)
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    The required dependencies for the experiment are listed in `pyproject.toml`. Install them using `uv`:
    ```bash
    uv pip install -e .[experiment]
    ```
    The `-e` flag installs the project in "editable" mode, and `[experiment]` installs the specific dependencies for the experiment group.

## Data

The experiment relies on a benchmark dataset which is not included in this repository.

- **Dataset:** The code is configured to use the BODMAS dataset.
- **Path:** Please place the dataset files inside the `data/bodmas/` directory. The structure should look like this:
  ```
  data/
  └── bodmas/
      ├── bodmas.npz
      └── bodmas_metadata.csv
  ```

## Usage
TBD

### Running the Experiment

The main entry point for running the experiment is `src/experiment/main.py`.

To execute the full experimental pipeline, run:
```bash
python src/experiment/main.py
```

### Generating Graphs

To generate the graphs and figures for the thesis, you can run the scripts located in the `src/graphs/` directory. The main script is `src/graphs/main.py`:
```bash
python src/graphs/main.py
```

## Running Tests

The project includes a suite of tests to verify the correctness of the data loading and processing logic. To run the tests, execute `pytest` from the root of the repository:
```bash
pytest
```
