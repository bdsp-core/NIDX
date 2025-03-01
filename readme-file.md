# Machine Learning Model for Clinical Text Analysis

## Overview
This repository contains code for training, evaluating, and testing a machine learning model for clinical text analysis. The model uses natural language processing (NLP) techniques to analyze clinical notes and predict binary outcomes based on textual features.

## Dependencies
- Python
- NumPy
- Pandas
- scikit-learn
- XGBoost (implied by the code structure)
- Matplotlib
- tqdm

## Project Structure

### Data Files
- `bidmc_notes_deid.csv`: Deidentified clinical notes from Beth Israel Deaconess Medical Center
- `annot_20240223.csv`: Annotation file with labels for classification
- Various pickle files containing trained models and preprocessed data:
  - `features_bow_2231227_manualbow_l1_c10.pkl`
  - `non_zero_features_2231227_manualbow_l1_c10.pkl`
  - `vectorizer_2231227_manualbow_l1_c10.pkl`
  - `model4_2231227_manualbow_l1_c10.pkl`
  - `X_train_l1_selected_2231227_manualbow_l1_c10.pkl`
  - `X_test_l1_selected_2231227_manualbow_l1_c10.pkl`
  - `y_train_2231227_manualbow_l1_c10.pkl`
  - `y_test_2231227_manualbow_l1_c10.pkl`

### Code Modules
- `load_data_preprocess.py`: Contains functions for loading and preprocessing text data
- `plots.py`: Contains visualization functions for model evaluation

## Functionality

### Text Preprocessing
- Text cleaning: Removing carriage returns and newlines
- Text matching with regular expressions
- Custom text preprocessing via the `preprocess_text` function

### Model Evaluation
The codebase includes robust evaluation functions:

1. `bootstrap_and_plot`: Performs bootstrap resampling to generate confidence intervals for model performance metrics and plots ROC and precision-recall curves.

2. `plot_top_feature_importances`: Visualizes the most important features in the trained model based on gain.

3. `plot_roc_curve`, `plot_precision_recall_curve`, `plot_roc_pr_curves`: Functions for visualizing model performance.

4. `bootstrap_and_plot_results`: An enhanced evaluation function that calculates and plots:
   - ROC curves with AUROC and confidence intervals
   - Precision-recall curves with AUPRC and confidence intervals
   - Additional metrics including F1 scores, recall, precision, and specificity for both classes
   - Confidence intervals for all metrics

### Model Testing
The code includes evaluation on two separate datasets:
1. MGB dataset (Massachusetts General Brigham): A limited subset of the main test data
2. BIDMC dataset (Beth Israel Deaconess Medical Center): External validation data

## Workflow
1. Load preprocessed data and trained model from pickle files
2. Preprocess clinical notes from the BIDMC dataset
3. Select specific indices from the MGB dataset for evaluation
4. Apply label corrections to both datasets (as of June 6, 2024)
5. Evaluate model performance on both datasets using bootstrap resampling

## Usage
The code is structured in blocks that can be executed sequentially:

1. **Block 1**: Load all the necessary pickle files containing the trained model and preprocessed data
2. **Block 2**: Define evaluation functions for model performance
3. **Block 3**: Load and preprocess the BIDMC dataset
4. **Block 4**: Define a limited subset of the MGB dataset for evaluation
5. **Block 5**: Apply label corrections to both datasets
6. **Block 6**: Evaluate model performance on the MGB dataset
7. **Block 7**: Transform the BIDMC dataset using the same vectorizer
8. **Block 8**: Evaluate model performance on the BIDMC dataset

## Notes
- The code suppresses warnings at the beginning of execution
- The model appears to be an XGBoost classifier with L1 regularization
- Feature selection has been applied to reduce dimensionality
- Bootstrap resampling with 1000 iterations is used to establish confidence intervals for performance metrics
- Recent label corrections (dated June 6, 2024) have been applied to both datasets

## Performance Metrics
The code evaluates model performance using:
- AUROC (Area Under the Receiver Operating Characteristic curve)
- AUPRC (Area Under the Precision-Recall Curve)
- F1 score, precision, recall, and specificity for both classes
- 95% confidence intervals for all metrics through bootstrap resampling
