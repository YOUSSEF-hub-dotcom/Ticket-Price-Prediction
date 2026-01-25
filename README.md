# ✈️ Flight Ticket Price Prediction

An **end-to-end Machine Learning project** that predicts flight ticket prices using structured data, advanced feature engineering, and a full **MLOps lifecycle with MLflow**.
The project is built with a **Data Scientist / ML Engineer mindset**, focusing on clean pipelines, explainability, reproducibility, and production readiness.

---

## 🚀 Project Overview

The objective of this project is to **predict flight ticket prices** based on multiple factors such as:

* Airline
* Route (Source → Destination)
* Number of stops
* Departure & arrival times
* Flight duration
* Travel date & seasonality

The solution covers the complete ML lifecycle:
**Data Cleaning → EDA → Feature Engineering → Modeling → Hyperparameter Tuning → MLflow Tracking → Model Registry → Production Gate**.

---

## 🧠 Project Architecture

```
Flight_Ticket_Price_Prediction/
│
├── data.py                   # Data cleaning & feature engineering
├── eda_visualization.py      # EDA & business-driven analysis
├── model.py                  # Model training & tuning (XGBoost)
├── MLflow_LifeCycle.py       # MLflow tracking & Model Registry
├── main.py                   # Pipeline entry point
│
├── MLproject                 # MLflow project configuration
├── conda.yaml                # Conda environment
└── README.md
```

---

## 📊 Dataset Description

* **Format**: Excel (.xlsx)
* **Target Variable**: `Price`
* **Key Columns**:

  * Airline, Source, Destination
  * Total_Stops
  * Dep_Time, Arrival_Time
  * Duration
  * Date_of_Journey

---

## 🧹 Data Cleaning & Preprocessing

Key preprocessing steps:

* Converted `Date_of_Journey` to datetime and extracted:

  * Day, Month, Day of Week, Quarter
* Extracted hour & minute from departure and arrival times
* Converted flight duration into **total minutes**
* Mapped `Total_Stops` to numerical values
* Removed redundant & low-value columns:

  * Route, Additional_Info, raw time columns
* Removed **duplicate rows** to prevent data leakage
* Handled missing values using **mode imputation**

---

## 📐 Feature Engineering

New features created to capture real-world flight behavior:

* `Dep_Session` (Early Morning / Morning / Evening / Night)
* `Is_Long_Flight` (Duration > 8 hours)
* `is_weekend`
* `is_peak_season`
* `Path` (Source-Destination)

### 📉 Target Transformation

* Detected **right skewness** in ticket prices
* Applied **log transformation** (`log1p`) to stabilize variance
* Removed price outliers using **IQR method**

---

## 🔍 Exploratory Data Analysis (EDA)

Business-driven questions answered:

* How do prices vary across airlines?
* Do more stops increase ticket prices?
* Are morning flights more expensive than night flights?
* Does seasonality affect pricing?
* How does flight duration correlate with price?

Visualizations used:

* Histograms & KDE plots
* Boxplots & violin plots
* Correlation analysis
* Heatmaps & trend analysis

---

## 🤖 Model Training

### Pipeline Components

* **Categorical Features** → OneHotEncoder
* **Numerical Features** → Passed directly
* **Model** → XGBoost Regressor

### Training Strategy

* Train/Test split: **80/20**
* 5-Fold Cross Validation for stability
* Sample weighting to improve generalization
* Hyperparameter tuning using:

  * RandomizedSearchCV
  * GridSearchCV

---

## 📈 Model Performance

| Metric        | Value      |
| ------------- | ---------- |
| R² Score      | **0.863**  |
| MAE           | **1404**   |
| RMSLE         | **0.192**  |
| CV MAE (mean) | **0.129**  |
| CV MAE (std)  | **0.0028** |

✅ Low CV variance indicates a **stable and reliable model**.

---

## 🧪 Experiment Tracking with MLflow

Tracked artifacts and metadata:

* Hyperparameters (Random + Grid Search)
* Evaluation metrics
* Feature importance
* Model artifacts
* Input/output signature

### 🔄 MLflow Lifecycle

1. Experiment Tracking
2. PyFunc Model Wrapping
3. Model Signature & Input Schema
4. Model Registry
5. Automated Quality Gate

### 🚦 Production Gate

```python
if r2 >= 0.85 and rmsle <= 0.20:
    → Production 🚀
else:
    → Staging 🛑
```

---

## 📦 Model Packaging

* Model logged as **MLflow PyFunc**
* Accepts structured DataFrame input
* Returns predicted ticket price in original scale

---

## ▶️ Run the Project

Using MLflow:

```bash
mlflow run . \
  -P data_path="path/to/Flight Ticket Price.xlsx" \
  -P n_estimators=1500 \
  -P max_depth=11 \
  -P learning_rate=0.05
```

---

## 🐍 Environment Setup

```yaml
python: 3.9
libraries:
- pandas
- numpy
- scikit-learn
- xgboost
- mlflow
- matplotlib
- seaborn
```

---

## 🎯 Key Takeaways

* Strong **feature engineering** drives performance
* Proper **EDA** leads to better modeling decisions
* MLflow enables **reproducibility & governance**
* Automated quality gates ensure **safe production deployment**

---

## 👨‍💻 Author

**Youssef Mahmoud**
Faculty of Computers & Information
Aspiring **Data Scientist / ML Engineer**

---

⭐ If you find this project useful, consider giving it a star on GitHub!


🔗 LinkedIn:
[https://www.linkedin.com/in/youssef-mahmoud-63b243361
](https://www.linkedin.com/in/youssef-mahmoud-63b243361?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B7LscMqrKS6S%2BXaYrgj7pPg%3D%3D)
⭐ If you like this project, consider giving it a star on GitHub!
