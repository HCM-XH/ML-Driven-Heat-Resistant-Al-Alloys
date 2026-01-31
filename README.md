# Machine Learning–Assisted Design of High-Strength Heat-Resistant Cast Aluminum Alloys

Code accompanying the study:

**Machine Learning–Assisted Design and Experimental Validation of High-Strength Heat-Resistant Cast Aluminum Alloys**

---

## Abstract

This repository provides the machine learning modeling and optimization framework developed for the data-driven design of high-strength, heat-resistant cast aluminum alloys. The approach integrates Random Forest regression for mechanical property prediction, exhaustive feature selection with cross-validation, and genetic algorithm–based global optimization of alloy compositions. Together, these methods enable efficient exploration of high-dimensional composition–process space and support accelerated alloy discovery.

---

## Repository Structure

├── Model.py
├── ga_alloy_design.py
└── README.md

### Model Construction (`Model.py`)

This script implements the development of the predictive model for tensile strength. The workflow includes loading experimentally derived alloy datasets, removing highly correlated descriptors, performing exhaustive feature combination search, selecting the optimal feature subset based on cross-validated R², training the final Random Forest regressor, evaluating predictive performance, and exporting the trained model for downstream optimization.

### Genetic Algorithm Optimization (`ga_alloy_design.py`)

This script performs compositional optimization using a genetic algorithm guided by the trained surrogate model. The framework enables continuous exploration of alloying elements within constrained design spaces, parallelized fitness evaluation, and population-based global search to maximize predicted tensile strength. The resulting candidates are intended to support experimental validation and accelerated alloy development.

---

## Computational Workflow

1. Train the machine learning model:

```bash
python Model.py

2. Run the genetic algorithm optimization:
python ga_alloy_design.py

3. Dependencies
Python ≥ 3.8
4. Required packages:
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
deap
openpyxl
Reproducibility Notes

Update file paths to match the local computing environment before execution.

Parallel computation may require substantial CPU resources.

Large genetic algorithm populations can significantly increase computational cost.
Intended Use

This repository is provided to support research reproducibility, methodological transparency, and further development of machine learning–guided alloy design strategies. The code is primarily intended for academic research.





