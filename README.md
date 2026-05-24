# Wine Quality Grader

This is an AI Project developd for the subject **TC3002B**.

The objective of this project is to predict wine quality based on physicochemical properties using machine learning techniques.

---

# Dataset

The dataset used in this project is the **Wine Quality Dataset** from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/186/wine+quality

Only the **white wine dataset** was selected due to its larger number of instances.

---

# Project Structure

### 🔹 Dataset

Contains the original dataset and the generated train/test splits.

* **test/**
    * `test.data` → No preprocessed test dataset
* **train/**
    * `train.data` → Preprocessed and augmented training dataset
* **winequality-white.csv** → Original dataset

---

### 🔹 Preprocessing

Folder where preoprocessing is made:
* `winequiality_tuning_dataset.ipynb`
    * Loads the dataset
    * Splits the data into training and testing sets
    * Standardizes numerical features
    * Applies data augmentation using SMOTE
    * Saves the processed datasets into their corresponding folders

---

# Evaluation Metrics

The performance of the models will be evaluated using the following metrics:

* **Accuracy**  
  Measures the proportion of correctly classified instances.

* **Precision**  
  Measures how reliable the predictions for each class are.

* **Recall**  
  Measures how many relevant instances were correctly identified.

* **F1-Score**  
  Measures the balance between Precision and Recall.
