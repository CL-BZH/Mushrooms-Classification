
---

# Mushroom Classification Project

## Overview

This project focuses on classifying mushrooms as **edible** or **poisonous** using machine learning techniques. The dataset used in this project contains 8,124 instances with 22 categorical attributes. The goal is to build models that can accurately distinguish between edible and poisonous mushrooms, with a particular emphasis on achieving **perfect precision** (i.e., no false positives).

The project is divided into two main Jupyter notebooks:
1. **`mushrooms_classification.ipynb`**: Focuses on traditional machine learning models such as Random Forest, Decision Tree, and XGBoost.
2. **`mushrooms_classification_with_mlp.ipynb`**: Explores the use of a Multilayer Perceptron (MLP) neural network to achieve high precision.

The project also includes a utility file (`tools.py`) containing all the functions used in the notebooks.

---

## Directory Structure

```
.
├── Data
│   └── mushroom
│       ├── agaricus-lepiota.data          # Mushroom dataset
│       └── agaricus-lepiota.names         # Dataset description
├── Notebooks
│   ├── mushrooms_classification.html      # HTML export of the first notebook
│   ├── mushrooms_classification.ipynb     # Jupyter notebook for traditional models
│   ├── mushrooms_classification_with_mlp.html  # HTML export of the second notebook
│   └── mushrooms_classification_with_mlp.ipynb # Jupyter notebook for MLP model
└── Utils
    └── tools.py                           # Utility functions for data processing and analysis
```

---

## Dataset

The dataset used in this project is the **Mushroom Classification Dataset**, available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom). It contains the following key characteristics:
- **Number of Instances**: 8,124
- **Number of Attributes**: 22 (all categorical)
- **Target Variable**: `class` (edible or poisonous)
- **Missing Values**: 2,480 missing values in the `stalk_root` attribute.

The dataset is stored in the `Data/mushroom` directory as `agaricus-lepiota.data` and `agaricus-lepiota.names`.

---

## Methodology

### Data Preprocessing
The preprocessing steps include:
1. **Handling Missing Values**: The `stalk_root` column was dropped due to a large number of missing values.
2. **Removing Non-Informative Features**: The `veil_type` column was dropped since it has a unique value for all samples.
3. **Data Transformation**:
   - **One-Hot Encoding**: Applied to categorical features.
   - **Target Encoding**: Encoded `e` (edible) as `1` and `p` (poisonous) as `0`.
4. **Feature Selection**: Focused on visually observable attributes.

### Model Training
Several models were trained and evaluated, including:
- **Naive Bayes**
- **Random Forest**
- **Decision Tree**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **XGBoost**
- **Multilayer Perceptron (MLP)**

### Feature Importance Analysis
Feature importance was analyzed to identify the most relevant features for classification. Techniques such as **Recursive Feature Elimination (RFE)**, **Mutual Information**, and **Chi-Square Test** were used for feature selection.

---

## Results

### Original Dataset
All models achieved **perfect precision** on the original dataset, with the Naive Bayes model being the simplest yet effective.

### Visual Features Only
When only visual features were used (excluding `odor`, `population`, and `habitat`), the **MLP model** outperformed traditional models in terms of precision, making it suitable for practical applications where false positives must be avoided.

---

## Tools and Functions

The `tools.py` file contains utility functions for data processing, model training, and evaluation. Key functions include:
- **`process_models`**: Processes a list of machine learning models and evaluates their performance.
- **`analyze_models`**: Analyzes and visualizes model performance and feature importances.
- **`plot_roc_curves`**: Plots ROC curves for multiple models.
- **`perform_rfe`**: Performs Recursive Feature Elimination (RFE) for feature selection.
- **`calculate_mutual_info`**: Calculates Mutual Information scores between features and the target variable.
- **`calculate_chi2_for_features`**: Performs Chi-Square Test for feature selection.

---

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:CL-BZH/Mushrooms-Classification.git
   cd Mushrooms-Classification
   ```

2. **Install Dependencies**:
   Ensure you have the required Python packages installed. You can install them using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebooks**:
   - Open `Notebooks/mushrooms_classification.ipynb` to explore traditional machine learning models.
   - Open `Notebooks/mushrooms_classification_with_mlp.ipynb` to explore the MLP model.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom).
- Special thanks to the authors of the dataset and the open-source community for their contributions.

---
