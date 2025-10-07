# bachelor-thesis-wellbeing-prediction

#Exploring Automated Machine Learning and Deep Learning for Well-being Prediction Using Harmonized Open-Source Datasets
1. Project Overview

This repository contains the full implementation developed for the bachelor thesis
“Prediction of Human Well-being using Physiological Signals (EDA and ECG)”.
The work aims to predict individual well-being levels based on electrodermal activity (EDA) and electrocardiogram (ECG) signals by combining traditional machine learning (AutoML) and deep learning approaches.

The system is organized into distinct modules for data aggregation, preprocessing, feature extraction, model training, and evaluation.
All scripts are written in Python and designed to enable reproducibility of the experimental results presented in the thesis.
2. The workflow operates as follows:

Features are generated using the NeuroKit2 toolbox within the agg_data_prepare stage.

The resulting feature tables are loaded into the AutoML pipeline.

The system explores preprocessing operators, feature transformations, and a range of learners (e.g., logistic regression, SVM, random forest, XGBoost).

Evaluation is performed using stratified five-fold cross-validation and, where applicable, leave-one-subject-out validation.

Performance metrics including Accuracy, F1-score, ROC-AUC, and Precision are saved under the results/ directory.


3. Processing Pipeline
3.1 Data Preparation and Feature Extraction

Raw datasets  were cleaned, harmonized, and converted into aggregated physiological features using the scripts in agg_data_prepare/.
To execute the complete extraction process:
This step:

Loads raw EDA and ECG signals.

Applies data cleaning, signal correction, and feature extraction.

Stores aggregated features in the agg_data/ directory.

These features serve as input for the NaiveAutoML classifiers

3.2 Naive AutoML 
find_classifier_single.py
find_classifier.py
test_classifier.py

3.3 Deep Learning Pipeline

For end-to-end modeling directly from time-series data, the repository includes LSTM and CNN architectures located in the deep_learning/ module.
Before model training, signal-level preprocessing must be performed:
cd prepare
python main.py
This stage:

Applies resampling, filtering, and z-score normalization.

Segments signals into fixed-length overlapping windows.

Produces NumPy arrays suitable for deep learning input.

Install dependencies via:

pip install -r requirements.txt
