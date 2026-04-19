# Laptop Price Predictor Pro

A professional, production-ready Streamlit application that predicts laptop prices using a Stacking Ensemble regression model.

## Overview

- **Data Science**: Exploratory Data Analysis (EDA), cleaning, and feature engineering in Jupyter Notebook.
- **Modeling**: Advanced Stacking Ensemble (XGBoost + Random Forest) for high-precision valuation.
- **Deployment**: Responsive Streamlit UI with real-world hardware constraints.

## 📊 Model Performance

The current production model is a **Stacking Regressor** (XGBoost + Random Forest) trained on a cleaned dataset of 1,300+ laptops.

- **R² Score**: `0.882` (Explains 88% of price variance)
- **Mean Absolute Error (MAE)**: `0.15` (Approx. 15% error margin on market value)
- **Methodology**: Log-transformation of target prices to handle heteroscedasticity across budget and premium brackets.

## 🛠️ Features

- **Intelligent Hardware Constraints**:
  - RAM options dynamically filter based on CPU power (e.g., Celeron/Pentium limited to 8GB).
  - Operating Systems locked to brand logic (e.g., macOS only for Apple).
  - Storage options restricted by device type (e.g., Ultrabooks use SSD exclusively).
- **Premium UI/UX**:
  - **Glassmorphism**: Semi-transparent sleek containers with backdrop filters.
  - **Visual Hierarchy**: Emerald-accented headers and clear focal points.
  - **Micro-interactions**: Smooth entrance animations and hover state transitions.
- **Advanced Engineering**: Automatic Pixels Per Inch (PPI) calculation from resolution and screen size.

## 📁 Project Structure

```
.
├── app.py              # Main Streamlit application
├── data/
│   └── laptop_data.csv # Raw dataset
├── models/             # Serialized model artifacts
│   ├── df.pkl          # Reference dataframe for UI options
│   └── pipe.pkl        # Trained Stacking Ensemble pipeline
├── laptop_data.ipynb   # Model training & EDA notebook
└── requirements.txt    # Project dependencies
```

## 🚀 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd Laptop-Price-Predictor
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## 📝 Usage Note
The app requires `df.pkl` and `pipe.pkl` in the `models/` directory. If they are missing, please re-run the final cells of `laptop_data.ipynb` to regenerate the artifacts.
