# Domain Specificity Benchmark

This anonymised repository accompanies our submission to ACL ARR 2025 (February) for the paper ``Extinction or Evolution? An Evaluation Framework for Linguistic Nuance, Historical Variation, and Domain-Specificity in the age of LLMs''

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#license)
## Introduction

Understanding how models perform across different domains, variations and nuances is crucial for developing robust and generalizable algorithms. 
This code aims to provide standardized evaluation metrics to assess linguistic nuance, historical variation and specificity capability. 

## Repository Structure

- `corpus/`: Contains datasets used for the experiments.
- `src/`: Source code for preprocessing,synonyms generations, evaluations and utilities.
- `results/`: Directory to store results.
- `requirements.txt`: Python dependencies required to run the code.

## Installation

To set up the environment, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vs1rr/domain_specificity_benchmark.git
   cd domain_specificity_benchmark
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the experiments, follow these steps:

1. **Create API keys** for GPT and Merriam-Webster Thesaurus.
2. **Pre-process the corpus** using the scripts in the `corpus_processing/` folder.
3. **Generate synonyms** using the code in the `syn_generation/` folder.
4. **Evaluate the results** by running the scripts in the `evaluation/` folder.

The benchmark implementation is available in:
```
src/evaluation/benchmark.py
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements 

The corpus used for our experiments was taken by these two works : 

- Maddela, M., & Xu, W. (2018, January). A Word-Complexity Lexicon and A Neural Readability Ranking Model for Lexical Simplification. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.
- Colliaux, D., & van Trijp, R. (2024). The discourse of the French method: making old knowledge on market gardening accessible to machines and humans.
