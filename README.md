# Aero-Health: Predictive Maintenance for Turbofan Engines

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Active-green)

## ✈️ Project Overview

**Aero-Health** is a Deep Learning project designed to predict the **Remaining Useful Life (RUL)** of turbofan jet engines. By analyzing high-frequency sensor telemetry data, this system enables proactive maintenance strategies, reducing unplanned downtime and preventing catastrophic engine failures.

The core of the project is an **LSTM (Long Short-Term Memory)** neural network that learns temporal degradation patterns from multi-variate time-series data.

## 📊 Dataset

We use the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset, specifically the **FD001** subset.

- **Data Source:** [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- **Inputs:** 21 sensor readings (Temperature, Pressure, Fan Speed, etc.) + 3 operational settings.
- **Target:** Remaining Useful Life (RUL) in cycles.
- **FD001 Characteristics:**
  - Single operating condition (Sea Level).
  - Single fault mode (HPC Degradation).

## 🛠️ Methodology

1.  **Data Preprocessing:**
    -   **RUL Calculation:** Computed RUL for training data based on max cycles.
    -   **Piecewise RUL:** Capped RUL at 125 cycles to focus learning on the degradation phase.
    -   **Feature Selection:** Removed constant and non-informative sensors.
    -   **Normalization:** Scaled all features to [0, 1] using Min-Max Scaling.

2.  **Sequence Generation:**
    -   Created sliding time windows of **50 cycles** to capture temporal context.
    -   Input shape: `(Samples, 50, Features)`.

3.  **Model Architecture:**
    -   **Layer 1:** LSTM (100 units) + Dropout (0.2)
    -   **Layer 2:** LSTM (50 units) + Dropout (0.2)
    -   **Output:** Dense layer (1 unit) for regression.

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Aero-Health.git
    cd Aero-Health
    ```
2.  Install dependencies:
    ```bash
    pip install numpy pandas matplotlib scikit-learn tensorflow
    ```

### Running the Project
Open the Jupyter Notebook to explore the analysis and training process:
```bash
jupyter notebook predictive_maintenance.ipynb
```

## 📈 Results

The model evaluates performance on the test set using **Root Mean Squared Error (RMSE)**.

- **Performance Metric:** RMSE (Cycles)
- **Visualizations:**
    -   Training History (Loss & MAE)
    -   True vs. Predicted RUL comparison
    -   Prediction Error Distribution
 
**Model loss function vs Epochs graph**
<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/37fb9916-7390-412b-900a-6b56752ed095" />

**Model Testing**
<img width="1141" height="528" alt="image" src="https://github.com/user-attachments/assets/f3553af4-9ec3-437d-8b31-d2b3826cb20e" />

## 📂 Project Structure

```
Aero-Health/
├── CMAPSSData/                 # Raw dataset files
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── predictive_maintenance.ipynb # Main Jupyter Notebook
├── predictive_maintenance.py    # Python script version
├── rul_prediction_model.h5      # Trained model file
├── feature_scaler.pkl           # Saved scaler object
└── README.md                    # Project documentation
```
