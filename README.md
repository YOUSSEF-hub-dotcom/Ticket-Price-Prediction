🎟️ Flight Ticket Price Prediction Project

A complete end-to-end Machine Learning project for predicting flight ticket prices, starting from raw data cleaning & advanced EDA, through feature engineering and model training, and ending with full MLOps lifecycle using MLflow, including model registry and production gating.

This project is built with a Data Scientist / ML Engineer mindset, focusing on:

Robust preprocessing

Strong evaluation strategy

Hyperparameter optimization

Reproducibility & production readiness

🚀 Project Overview

The goal of this project is to predict flight ticket prices accurately based on multiple factors such as airline, route, duration, departure time, number of stops, and seasonality.

The solution includes:

Advanced data cleaning & preprocessing

Deep Exploratory Data Analysis (EDA)

Feature engineering driven by domain knowledge

XGBoost regression with hyperparameter tuning

Cross-validation for stability

MLflow experiment tracking & model registry

Automatic quality gate for Production vs Staging

🧠 Project Architecture
Flight_Ticket_Price_Prediction/
│
├── data.py                  # Data loading, cleaning & feature engineering
├── eda_visualization.py     # EDA & visual analytics
├── model.py                 # Model training & hyperparameter tuning
├── MLflow_LifeCycle.py      # MLflow tracking, registry & quality gate
├── main.py                  # End-to-end pipeline entry point
│
├── MLproject                # MLflow Project configuration
├── conda.yaml               # Conda environment
├── README.md

📊 Dataset

Source: Flight Ticket Price Dataset (Excel format)

Target Variable:

Price → Flight ticket price (log-transformed during training)

Key Features:

Airline

Source / Destination

Total Stops

Departure & Arrival Time

Flight Duration

Journey Date (Month, Day, Weekday, Quarter)

🧹 Data Cleaning & Preprocessing

Key steps applied:

Datetime parsing (Date_of_Journey)

Time feature extraction (hours & minutes)

Duration conversion to minutes

Mapping categorical stops to numeric values

Dropping redundant & low-value columns

Removing duplicated rows (222 rows)

Handling missing values

Log transformation of target to treat right skew

Outlier detection using IQR (removed extreme prices)

📌 Result: Clean, stable, and model-ready dataset

🔍 Exploratory Data Analysis (EDA)

EDA focused on answering real business questions, such as:

How do prices vary across airlines?

Are non-stop flights always more expensive?

Does duration strongly affect ticket price?

Are weekend or peak-season flights pricier?

How do route & season interact?

Visualizations include:

Histograms & boxplots

Violin plots

Scatter & regression plots

Pivot tables & heatmaps

Multi-factor interaction analysis

🧠 Feature Engineering

New features created:

Duration_mins

Dep_hour, Arrival_hour

Day_of_Week, Month_of_Journey, Quarter

is_weekend

is_peak_season

Dep_Session (Early Morning → Night)

Is_Long_Flight

Path (Source → Destination)

📌 All transformations are reproducible and pipeline-safe.

🤖 Model Training

Problem Type: Regression

Model Used:

XGBoost Regressor (XGBRegressor)

Pipeline:

OneHotEncoding for categorical features

Numerical features passed directly

End-to-end sklearn Pipeline

Validation Strategy:

Train / Test split: 80% / 20%

K-Fold Cross Validation (5 folds)

Sample weighting to handle price scale

Hyperparameter Tuning:

RandomizedSearchCV (broad exploration)

GridSearchCV (fine-tuning)

📈 Model Performance
Metric	Value
CV MAE (Mean)	0.1297
CV MAE (Std)	0.0029
MAE (Actual Price)	1404.36
R² Score	0.8633
RMSLE	0.1923

📌 Metrics were computed on original price scale after inverse log transformation.

🧪 Experiment Tracking with MLflow

Tracked using MLflow:

Hyperparameters

Cross-validation metrics

Final evaluation metrics

Feature importance

Model artifacts

Input examples & model signature

🔁 MLflow Lifecycle & Quality Gate

Model Registry Flow:

Log model as MLflow PyFunc

Register model → TicketPricePredictor

Move to Staging

Apply Quality Gate

🚦 Quality Gate Rules
R2 ≥ 0.85
RMSLE ≤ 0.20


✅ Passed → Production 🚀

❌ Failed → Staging 🛑

📦 This ensures only high-quality models reach production.

📦 Model Packaging

Wrapped as MLflow PyFunc

Framework-agnostic

Accepts pandas DataFrame

Outputs predicted ticket price (original scale)

⚙️ MLflow Project

Run the entire pipeline using:

mlflow run . \
  -P n_estimators=1500 \
  -P max_depth=11 \
  -P learning_rate=0.05

🐍 Environment Setup
name: ticket_price_env
python: 3.9
libraries:
- mlflow
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost

🎯 Key Takeaways

Strong EDA-driven feature engineering

Robust regression modeling

Proper evaluation on transformed targets

Real MLOps lifecycle (not just training)

Production-quality ML project structure

👨‍💻 Author

Youssef Mahmoud
Faculty of Computers & Information
Aspiring Data Scientist / ML Engineer

🔗 LinkedIn:
[https://www.linkedin.com/in/youssef-mahmoud-63b243361
](https://www.linkedin.com/in/youssef-mahmoud-63b243361?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B7LscMqrKS6S%2BXaYrgj7pPg%3D%3D)
⭐ If you like this project, consider giving it a star on GitHub!
