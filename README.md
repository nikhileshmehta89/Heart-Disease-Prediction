# Heart Disease Prediction: Deep Learning Case Study

This notebook demonstrates a binary classification workflow for predicting heart disease using a small feed-forward neural network. It builds and evaluates one model from scratch with NumPy and a comparable model with TensorFlow/Keras.

## Dataset

The notebook is designed for Google Colab and prompts you to upload a CSV file. The dataset should include a binary `target` column and numeric predictor columns, including `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`.

The target is interpreted as `0` for no heart disease and `1` for heart disease.

## Run the Notebook

1. Open the notebook in Google Colab.
2. Run the setup and import cells.
3. Run the dataset-upload cell and select the CSV file.
4. Run the remaining cells from top to bottom.

The notebook uses NumPy, pandas, Matplotlib, Seaborn, scikit-learn, and TensorFlow/Keras. If a package is missing, install it in the notebook environment before continuing.

## Workflow

- **Explore the data:** Preview its dimensions, data types, sample rows, summary statistics, and missing-value counts. Plot target balance and selected feature distributions.
- **Check data quality:** Use the IQR rule to flag potential outliers in selected clinical measurements. The notebook retains flagged values because they may be valid observations.
- **Prepare the data:** Create stratified training, validation, and test splits of approximately 60%, 20%, and 20%. Fit `StandardScaler` on the training set only, then apply it to the validation and test sets to avoid leakage.
- **Train the NumPy model:** Implement the forward pass, sigmoid activation, binary cross-entropy loss, backpropagation, and gradient-descent updates.
- **Evaluate the NumPy model:** Report accuracy, precision, recall, F1, and ROC AUC, and display a confusion matrix and ROC curves.
- **Train the Keras model:** Build a similar network using Keras layers, Adam, binary cross-entropy, and early stopping.
- **Compare configurations:** Sweep learning rates, hidden-layer sizes, and optimizers. Use validation results to select a configuration; reserve the test set for final evaluation.

## Results From One Run

In one recorded run, the NumPy model achieved **82.93% test accuracy** and **93.44% ROC AUC**. The Keras model achieved **84.39% test accuracy** and **92.19% ROC AUC**.

These values are specific to that run and may vary with the dataset, environment, or training configuration. Accuracy and ROC AUC measure different aspects of performance, so the higher accuracy does not by itself mean one model is better overall.

## Fairness and Limitations

Overall metrics can hide differences in model performance between demographic groups. The notebook does not currently calculate subgroup metrics. A useful next step is to compare recall and false-negative rates across age and sex groups, while reporting each group’s sample size.

This notebook is an educational exercise, not a clinically validated diagnostic tool. Its predictions should not be used to make medical decisions.
