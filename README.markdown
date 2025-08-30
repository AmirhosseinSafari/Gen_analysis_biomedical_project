# Amyloid Prediction Project

## Project Goal
The goal of this project is to develop a machine learning-based approach for predicting amyloidogenic regions in protein sequences by analyzing physicochemical features of amino acids. The project aims to classify protein sequences as amyloid-forming (positive) or non-amyloid-forming (negative) using a combination of feature extraction, sequence transformation, and classification models. This can aid in understanding protein aggregation associated with diseases like Alzheimer's and Parkinson's.

## Overall Process
The project follows a structured pipeline to achieve the classification goal:

1. **Data Preparation**:
   - Load amino acid physicochemical feature data and protein sequence datasets (training and test sets for positive and negative classes).
   - Select a subset of relevant physicochemical features for analysis.

2. **Feature Normalization**:
   - Normalize the selected features to ensure consistent scaling across different properties, improving model performance.

3. **Feature Labeling**:
   - Reduce the amino acid alphabet by clustering physicochemical features into discrete groups using k-means clustering, creating a simplified representation of amino acids.

4. **Sequence Transformation**:
   - Transform protein sequences into numerical feature vectors using a sequence graph transform (SGT) method, which captures pairwise relationships between amino acids based on the reduced alphabet.

5. **Feature Matrix Construction**:
   - Generate feature matrices for training and test sets by concatenating transformed sequence features for multiple physicochemical properties.

6. **Model Training and Evaluation**:
   - Train multiple machine learning models (Decision Tree, Linear SVM, RBF SVM, XGBoost, and k-Nearest Neighbors) on the training feature matrix.
   - Evaluate model performance on the test set using metrics such as accuracy, sensitivity, precision, F1-score, and Area Under the Precision-Recall Curve (AUPR).
   - Compare models to identify the best-performing configuration based on accuracy.

7. **Parameter Optimization**:
   - Iterate over different numbers of clusters (n) for the reduced alphabet and kernel parameters (k) for the SGT to find the optimal combination that maximizes classification performance.

8. **Result Storage**:
   - Save the best-performing model and its parameters for future use and reproducibility.

## Steps to Run the Project
1. **Setup Environment**:
   - Ensure MATLAB is installed with required toolboxes (Statistics and Machine Learning Toolbox, Bioinformatics Toolbox).
   - Clone the repository and place the required data files (`AmyPredData.mat`, `aaindex_all.mat`) in the project directory.

2. **Execute the Script**:
   - Run the main MATLAB script (`amynoacid_analysis.m`) to process the data, train models, and generate performance metrics.

3. **View Results**:
   - Check the console output for performance metrics tables for each parameter combination.
   - Review the saved `bestModel.mat` file for the top-performing model and its parameters.

## Dependencies
- MATLAB
- Statistics and Machine Learning Toolbox
- Bioinformatics Toolbox
- Input data files: `AmyPredData.mat`, `aaindex_all.mat`

## Notes
- The project assumes binary classification (amyloid vs. non-amyloid sequences).
- The performance metrics are displayed for each parameter combination, allowing easy comparison across models and settings.
- The saved model can be used for further analysis or deployment in amyloid prediction tasks.