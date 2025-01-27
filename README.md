# ethical-genai-dissertation

# Ethical Considerations in Text-Based Generative AI Models


## Project Overview

This repository contains the code and analyses from my dissertation titled **"Analysis of Ethical Considerations on Text-Based GenAI Models Using Artificial Intelligence."** The research investigates how biases manifest in text-based Generative AI systems and evaluates various classification algorithms across multiple datasets to assess their performance and fairness.

## Datasets

The analysis utilizes the following publicly available datasets:

1. **Adult Dataset**  
   - **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)  
   - **Description**: Predicts whether an individual's income exceeds \$50K/year based on census data.

2. **COMPAS Dataset**  
   - **Source**: [ProPublica](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)  
   - **Description**: Contains recidivism risk scores used in criminal justice, with documented racial biases.

3. **CrowS-Pairs Dataset**  
   - **Source**: [GitHub](https://github.com/nyu-mll/crows-pairs)  
   - **Description**: Tests model’s ability to distinguish stereotypical vs. anti-stereotypical sentence pairs.

4. **Hate Speech Dataset**  
   - **Source**: [Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)  
   - **Description**: Tweets labeled as hateful, offensive, or neutral.

5. **Jigsaw Dataset**  
   - **Source**: [Conversation AI](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)  
   - **Description**: Toxic online comments with identity-based attacks.

## Methodology

The research employs a mixed-methods approach, combining a systematic literature review with empirical evaluations of multiple classification algorithms across the selected datasets.

### Steps Involved:

1. **Data Preprocessing**:
   - **Numeric Datasets (Adult, COMPAS)**: Handled missing values, used one-hot encoding for categorical features, and applied feature scaling.
   - **Textual Datasets (CrowS-Pairs, Hate Speech, Jigsaw)**: Cleaned text (removing punctuation, lowercasing), tokenized, and used TF-IDF vectorization.

2. **Classification Algorithms**:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - Naive Bayes (Gaussian/Multinomial)

3. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrices & ROC/AUC analysis

4. **Bias Assessment**:
   - Analyzed performance disparities across demographic groups and content categories to identify potential biases.

## Usage

### Prerequisites

- **Python 3.8+**
- **Jupyter Notebook**
- **Required Python Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ethical-genai-dissertation.git
   cd ethical-genai-dissertation
