# NIDX - Code for: "A Machine Learning Approach for Identifying People with Neuroinfectious Diseases in Electronic Health Records"

## Overview
This repository contains code for training, evaluating, and testing a machine learning model that identifies neuroinfectious diseases (NIDX) from unstructured clinical notes. The model addresses limitations of traditional ICD code-based identification methods by leveraging natural language processing (NLP) techniques to analyze clinical text and accurately classify patients with NIDX.

## Dependencies
- Python
- NumPy
- Pandas
- scikit-learn
- XGBoost (implied by the code structure)
- Matplotlib
- tqdm

## Project Structure

### Background and Motivation
Neuroinfectious diseases (NIDX) pose serious threats to neurological health and may have long-term consequences, including increased risk for neurodegenerative diseases like Alzheimer's. Accurately identifying NIDX cases in electronic health records (EHR) is critical for clinical research but challenging due to:

- Imprecise identification using ICD billing codes (high sensitivity but low specificity)
- Labor-intensive manual chart reviews that are impractical for large datasets
- Diverse array of pathogens and often clinically indistinguishable symptoms

This project aims to overcome these limitations by using machine learning to analyze unstructured clinical notes, providing more accurate and efficient NIDX identification than traditional methods.

## Data Files
- `bidmc_notes_deid.csv`: Deidentified clinical notes from Beth Israel Deaconess Medical Center (BIDMC) used for external validation
- `annot_20240223.csv`: Annotation file with expert-labeled ground truth for classification
- Various pickle files containing trained models and preprocessed data:
  - `features_bow_2231227_manualbow_l1_c10.pkl`: Bag-of-Words features
  - `non_zero_features_2231227_manualbow_l1_c10.pkl`: Selected non-zero features after feature selection
  - `vectorizer_2231227_manualbow_l1_c10.pkl`: Text vectorizer model
  - `model4_2231227_manualbow_l1_c10.pkl`: Trained XGBoost model
  - `X_train_l1_selected_2231227_manualbow_l1_c10.pkl`: Training features
  - `X_test_l1_selected_2231227_manualbow_l1_c10.pkl`: Testing features
  - `y_train_2231227_manualbow_l1_c10.pkl`: Training labels
  - `y_test_2231227_manualbow_l1_c10.pkl`: Testing labels

### Code Modules
- `load_data_preprocess.py`: Contains functions for loading and preprocessing text data
- `plots.py`: Contains visualization functions for model evaluation

## Functionality

### Data Collection and Preprocessing
- **Source Data**: The model was developed using clinical notes from patients who underwent lumbar punctures at Mass General Brigham (MGB) hospitals, with external validation on notes from Beth Israel Deaconess Medical Center (BIDMC)
- **Ground Truth**: Six physician experts in neuroimmunology or neuroinfectious diseases manually labeled notes following a standardized operating procedure
- **Text Processing**:
  - Text cleaning: Removing carriage returns and newlines
  - Text matching with regular expressions
  - Custom text preprocessing via the `preprocess_text` function
  - Conversion to bag-of-words representation using n-grams (n=1, 2, 3)

### Feature Selection and Model Training
- From an initial set of 1,284 n-gram features, 342 were selected as significant predictors
- Feature selection was performed using Logistic Regression with L1 regularization
- Several models were evaluated through 5-fold cross-validation, with XGBoost selected for its superior performance, particularly on the Area Under the Precision-Recall Curve (AUPRC)
- The model was optimized to address class imbalance, as only 16% of notes in the training data represented positive NIDX cases

### Model Evaluation
The codebase includes robust evaluation functions:

1. `bootstrap_and_plot`: Performs bootstrap resampling to generate confidence intervals for model performance metrics and plots ROC and precision-recall curves.

2. `plot_top_feature_importances`: Visualizes the most important features in the trained model based on gain. The top features included clinically relevant terms such as "meningitis," "ventriculitis," and "meningoencephalitis."

3. `plot_roc_curve`, `plot_precision_recall_curve`, `plot_roc_pr_curves`: Functions for visualizing model performance.

4. `bootstrap_and_plot_results`: An enhanced evaluation function that calculates and plots:
   - ROC curves with AUROC and confidence intervals
   - Precision-recall curves with AUPRC and confidence intervals
   - Additional metrics including F1 scores, recall, precision, and specificity for both classes
   - Confidence intervals for all metrics using 1,000 bootstrap iterations

### Model Testing and Performance
The code includes evaluation on two separate datasets:
1. MGB dataset (Massachusetts General Brigham): A limited subset of the main test data (445 notes)
2. BIDMC dataset (Beth Israel Deaconess Medical Center): External validation data (600 notes)

Performance metrics:
- On the MGB test set:
  - AUROC: 0.977 (95% CI: 0.964-0.988)
  - AUPRC: 0.894 (95% CI: 0.831-0.943)
  - F1 score: 0.822 (95% CI: 0.752-0.879)
  - Recall: 0.846 (95% CI: 0.753-0.923)
  - Precision: 0.802 (95% CI: 0.709-0.889)
  - Specificity: 0.960 (95% CI: 0.939-0.978)

- On the BIDMC external validation set:
  - AUROC: 0.976 (95% CI: 0.961-0.989)
  - AUPRC: 0.779 (95% CI: 0.655-0.885)
  - F1 score: 0.658 (95% CI: 0.528-0.778)
  - Recall: 0.687 (95% CI: 0.538-0.839)
  - Precision: 0.637 (95% CI: 0.487-0.795)
  - Specificity: 0.976 (95% CI: 0.963-0.988)

The model significantly outperformed ICD-code based identification (which had a sensitivity of 97.1% but specificity of only 59.1%) and also surpassed a zero-shot LLaMA 3.2 model (which achieved an AUROC of 0.80 and specificity of 0.94, but lower recall of 0.64).

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

## Significance and Applications
- This NLP-based model enables accurate identification of neuroinfectious disease cases from clinical notes, addressing the limitations of ICD code-based methods
- The approach offers significant advantages for large-scale epidemiological research, particularly for studying associations between neuroinvasive pathogens and long-term outcomes like neurodegenerative diseases
- The model demonstrated strong performance across two independent hospital datasets, suggesting potential generalizability to other institutions
- The selected features align with clinical expertise, focusing on markers of CNS inflammation, diagnostic tests, and specific pathogens

## Notes
- The code suppresses warnings at the beginning of execution
- The model is an XGBoost classifier with feature selection performed using L1 regularization
- Original data included 3,000 notes from MGB, with 16% (479 notes) labeled as NIDX by expert review
- Bootstrap resampling with 1000 iterations is used to establish confidence intervals for performance metrics
- Recent label corrections (dated June 6, 2024) have been applied to both datasets to improve label accuracy
- Model is part of a research effort described in "A Machine Learning Approach for Identifying People with Neuroinfectious Diseases in Electronic Health Records" by Singh, Sartipi, et al.
