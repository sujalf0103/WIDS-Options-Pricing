# WIDS-Options-Pricing

# WIDS Options Pricing Project

This repository contains the coursework and final project submission for the **Winter in Data Science (WIDS)** initiative. The project focuses on financial modeling and options pricing using the **Asian Paints (`ASIANPAINT`)** dataset.

## 📂 Repository Structure

The repository is organized into two main sections:

### 1. Course Assignments
* **`Week 1/`**: Contains notebooks, code, and solutions for the Week 1 assignments.
* **`Week 2/`**: Contains notebooks, code, and solutions for the Week 2 assignments.

### 2. Final Project Submission
The files located in the **root directory** represent the **Final Project**. This submission implements and compares three different approaches to pricing options:
* **Benchmark Model**: A baseline statistical approach for comparison.
* **MLP (Multi-Layer Perceptron)**: A feedforward neural network model.
* **LSTM (Long Short-Term Memory)**: A recurrent neural network model optimized for time-series data.

---

## 🚀 Final Project Usage

Follow these instructions to run the Final Project code located in the root directory.

### Prerequisites
Ensure you have the necessary Python libraries installed:

```bash
pip install numpy pandas tensorflow scikit-learn openpyxl
```
Dataset
The project uses the ASIANPAINT_Dataset.xlsx file located in the root directory. Please ensure this file is present before running any scripts.

How to Run
Step 1: Train the Models
To train the deep learning models from scratch, run the following scripts. These will generate the .h5 model files.

```bash
# Train the MLP model
python train_mlp1.py

# Train the LSTM model
python train_LSTM.py
```

# Train the MLP model
python train_mlp1.py

# Train the LSTM model
python train_LSTM.py


```bash
# Predict using MLP
python predict_mlp1.py

# Predict using LSTM
python predict_LSTM.py
```

Step 3: Run Benchmark
To see the baseline performance for comparison:


```bash
python Benchmark.py
```

File Descriptions (Root Directory)
data_loader.py: Data preprocessing utility used by the MLP and Benchmark models.

data_loader_lstm.py: Data preprocessing utility formatted specifically for LSTM sequences.

train_mlp1.py: Script to train the Multi-Layer Perceptron model.

train_LSTM.py: Script to train the Long Short-Term Memory model.

predict_mlp1.py: Script to load the saved MLP model and make inferences.

predict_LSTM.py: Script to load the saved LSTM model and make inferences.

mlp1_model.h5 / lstm_model.h5: Saved weights for the trained models.


Author
GitHub: sujalf0103

Event: Winter in Data Science (WIDS)
