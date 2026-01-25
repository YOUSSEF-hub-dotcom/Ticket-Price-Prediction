import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

df = pd.read_excel("D:\ALL Projects\Ticket\Flight Ticket Price.xlsx" )
pd.set_option('display.width', None)
print(df.head(30))

print("=========== Basic Functions ==========")

print("information about data:")
print(df.info())

print("Statistical Operations:")
print(df.describe())

print("Columns:")
print(df.columns)

print("number of rows & columns:")
print(df.shape)

print("Column types:")
print(df.dtypes)

print("=========== Data Cleaning ==========")
#['Airline', 'Source', 'Destination', 'Arrival_Time', 'Duration',
#'Total_Stops', 'Additional_Info', 'Price', 'Date_of_Journey', 'Route', 'Dep_Time']

print("Validate DataType:")
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)

# extract hours and minute from  Dep_Time
df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format='%H:%M').dt.hour
df["Dep_minute"] = pd.to_datetime(df["Dep_Time"], format='%H:%M').dt.minute

# extract hours and minute from  Arrival_Time
df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"].str.split(' ').str[0], format='%H:%M').dt.hour
df["Arrival_minute"] = pd.to_datetime(df["Arrival_Time"].str.split(' ').str[0], format='%H:%M').dt.minute

# Convert Duration just to minutes
def convert_duration(duration):
    hours = 0
    minutes = 0

    parts = duration.split()

    for part in parts:
        if 'h' in part:
            hours = int(part.replace('h', ''))
        elif 'm' in part:
            minutes = int(part.replace('m', ''))

    return (hours * 60) + minutes

df["Duration_mins"] = df["Duration"].apply(convert_duration)

# Convert Total_Stops to Numeric
stop_dict = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
df['Total_Stops'] = df['Total_Stops'].map(stop_dict)

# After validate data type , now we don't Dep_Time &  Arrival_Time because we extract minutes and hours from them
df.drop('Dep_Time', axis=1, inplace=True)
df.drop('Arrival_Time', axis=1, inplace=True)

# and remove Duration because we extract it to minutes
df.drop('Duration', axis=1, inplace=True)

# Remove Route Column because we have Source & Destination Column
df.drop('Route', axis=1, inplace=True)

print(df['Additional_Info'].value_counts())
# Remove Additional_Info Because most of its values  No info > 80%
# Therefore, it will not provide educational value to the model and may cause Overfitting , High variance, Noise.
df.drop('Additional_Info', axis=1, inplace=True)

print("number of frequency rows")
print(df.duplicated().sum())

# we found 222 rows duplicated data this big problem (Overfitting ,Data Leakage ) so we must delete it
df.drop_duplicates(inplace=True)
print(f"Dataset shape after removing duplicates: {df.shape}")

print("missing values:")
print(df.isnull().sum())
print("----------")

imputer_mode = SimpleImputer(strategy='most_frequent') # mode
df["Total_Stops"] = imputer_mode.fit_transform(df[["Total_Stops"]])
print(df.isnull().sum())

sns.heatmap(df.isnull(), cmap="YlOrRd")
plt.title(" No Missing Values")
plt.show()

print(df.head(30))
print(df.dtypes)

print("=========== Data Preprocessing ==========")

# Price Skew
skew_value = df['Price'].skew()
print("Skew Value of Price:",skew_value)

sns.histplot(df['Price'],kde=True)
plt.title("Distribution of Price Before Treatment Skew")
plt.show()

# Skew value of Price = 1.8 ---> we have Right-Skewed
df['Price'] = np.log1p(df['Price'])

treat_skew_price = df['Price'].skew()
print("Treatment Skew of Price:",treat_skew_price)
# now it become : -2
sns.histplot(df['Price'],kde=True)
plt.title("Distribution of Price After Treatment Skew (Log Transformation)")
plt.show()

print("--------------------")

# Price Outlier
print(df['Price'].describe())
# We noticed a difference between the max &75%
#and the min &25% , which made us suspect there might be an outlier in the price column


Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)

IQR = Q3 - Q1
Lower = Q1 - 1.5 * IQR
Upper = Q3 + 1.5 * IQR

outliers = df[(df['Price'] < Lower) | (df['Price'] > Upper)]
print("Outliers Detect:",outliers)
print("Percentage of Outliers",len(outliers)/len(df) * 100,"%")
# Percentage of Outliers 0.08603383997705764 %

sns.boxplot(df['Price'],color='blue')
plt.title("BoxPlot to detect outlier in Price")
plt.show()

# we found 9 Rows in data contain outlier ----> So we Remove Them
df = df[(df['Price'] >= Lower) & (df['Price'] <= Upper)]
print(f"New Dataset shape: {df.shape}")

print("--------------------")

df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
df['Year_of_Journey'] = df['Date_of_Journey'].dt.year.nunique()
df['Month_of_Journey'] = df['Date_of_Journey'].dt.month
df['Days_of_Journey'] = df['Date_of_Journey'].dt.day
df['Day_of_Week'] = df['Date_of_Journey'].dt.weekday
df['Quarter'] = df['Date_of_Journey'].dt.quarter

df.drop("Date_of_Journey", axis=1, inplace=True)

if df['Year_of_Journey'].nunique() <= 1:
    df.drop("Year_of_Journey", axis=1, inplace=True)
    print("Year column removed because it has only one value.")

df['is_weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 4 else 0)

df['Path'] = df['Source'] + "-" + df['Destination']

def assign_session(hour):
    if (hour >= 4) and (hour < 8):
        return 'Early Morning'
    elif (hour >= 8) and (hour < 12):
        return 'Morning'
    elif (hour >= 12) and (hour < 16):
        return 'Noon'
    elif (hour >= 16) and (hour < 20):
        return 'Evening'
    elif (hour >= 20) and (hour < 24):
        return 'Night'
    else:
        return 'Late Night'

df['Dep_Session'] = df['Dep_hour'].apply(assign_session)

df['Is_Long_Flight'] = df['Duration_mins'].apply(lambda x: 1 if x > 480 else 0)

df['is_peak_season'] = df['Month_of_Journey'].apply(lambda x: 1 if x in [3, 5, 6, 12] else 0)

print(df.head(30))

print("=========== EDA & Visualization ==========")

print("What is the distribution of airfare prices?")
print(df['Price'].describe())

plt.figure(figsize=(8,5))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title("Distribution of Flight Ticket Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

print("--------------------------")
print("Does the ticket price vary depending on the airline?")
ticket_price_per_Airline = df.groupby('Airline')['Price'].agg(['mean', 'median', 'std']).sort_values(by='mean', ascending=False)
print(ticket_price_per_Airline)

plt.figure(figsize=(10,5))
sns.boxplot(x='Airline', y='Price', data=df,color='blue')
plt.title("Ticket Price Distribution by Airline")
plt.xticks(rotation=45)
plt.show()

print("--------------------------")
print("Are there airlines whose prices are noticeably higher than others?")
Avg_Price_Airline = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
print(Avg_Price_Airline)

plt.figure(figsize=(10,5))
Avg_Price_Airline.plot(kind='barh')
plt.title("Average Ticket Price per Airline")
plt.xlabel("Average Price")
plt.ylabel("Airline")
plt.show()

print("--------------------------")
print("Is the price difference within the same airline significant?")
STD_Price_Airline = df.groupby('Airline')['Price'].std().sort_values(ascending=False)
print(STD_Price_Airline)

plt.figure(figsize=(10,5))
sns.violinplot(x='Airline', y='Price', data=df)
plt.title("Price Variability Within Each Airline")
plt.xticks(rotation=45)
plt.show()

print("--------------------------")

print("Does the departure city(Source) affect the ticket price?")
Avg_Price_Source = df.groupby('Source')['Price'].mean().sort_values(ascending=False)
print(Avg_Price_Source)

plt.figure(figsize=(8,5))
Avg_Price_Source.plot(kind='bar',color='blue')
plt.title("Ticket Price by Source City")
plt.show()

print("--------------------------")
print("Does the arrival city(Destination) affect the ticket price?")
Avg_Price_Destination = df.groupby('Destination')['Price'].mean().sort_values(ascending=False)
print(Avg_Price_Destination)

plt.figure(figsize=(8,5))
sns.boxplot(x='Destination', y='Price', data=df,color='green')
plt.title("Ticket Price by Destination City")
plt.show()

print("--------------------------")
print("Does the ticket price vary depending on the number of stops?")
TotalStops_Price_of_Ticket = df.groupby('Total_Stops')['Price'].agg(['mean', 'median', 'std'])
print(TotalStops_Price_of_Ticket)

plt.figure(figsize=(8,5))
sns.barplot(x='Total_Stops', y='Price', data=df,color='Maroon')
plt.title("Ticket Price by Number of Stops")
plt.xlabel("Total Stops")
plt.ylabel("Price")
plt.show()

print("--------------------------")

print("Does the departure time(Dep_hour) affect the ticket price?")
Dep_hour_affectOnPrice = df.groupby('Dep_hour')['Price'].mean().head()
print(Dep_hour_affectOnPrice)

plt.figure(figsize=(10,5))
sns.lineplot(x='Dep_hour', y='Price', data=df, estimator='mean',color='coral')
plt.title("Average Price vs Departure Hour")
plt.gray()
plt.grid()
plt.show()

print("--------------------------")
print("Are morning flights more expensive than night flights?")
Deep_Session_per_price = df.groupby('Dep_Session')['Price'].mean().sort_values(ascending=False)
print(Deep_Session_per_price)

plt.figure(figsize=(8,5))
sns.barplot(x='Dep_Session', y='Price', color='teal' ,data=df)
plt.title("Average Price by Departure Session")
plt.show()

print("--------------------------")

print("Does the arrival time affect the price?")
Arrival_time_affectOnPrice = df.groupby('Arrival_hour')['Price'].mean().head()
print(Arrival_time_affectOnPrice)

plt.figure(figsize=(10,5))
sns.lineplot(x='Arrival_hour', y='Price', data=df, estimator='mean',color='purple')
plt.title("Average Price vs Arrival Hour")
plt.grid()
plt.gray()
plt.show()

print("--------------------------")
print('Does the duration of the trip affect the ticket price?')
print(df[['Duration_mins','Price']].corr())

plt.figure(figsize=(8,5))
sns.scatterplot(x='Duration_mins', y='Price', data=df, alpha=0.5)
plt.title("Price vs Flight Duration")
plt.show()

print("--------------------------")
print('Is the relationship between flight duration and price linear?')
print(df[['Duration_mins','Price']].corr(method='pearson'))

plt.figure(figsize=(8,5))
sns.regplot(x='Duration_mins', y='Price', data=df, scatter_kws={'alpha':0.3})
plt.title("Linear Relationship Between Duration and Price")
plt.show()

print("--------------------------")
print('Are long flights always more expensive than short flights?')
print(df.groupby('Is_Long_Flight')['Price'].mean())

plt.figure(figsize=(6,5))
sns.boxplot(x='Is_Long_Flight', y='Price', data=df,color='purple')
plt.title("Price Comparison: Long vs Short Flights")
plt.show()

print("--------------------------")

print('Does the ticket price vary depending on the day of the trip?')
print(df.groupby('Day_of_Week')['Price'].mean())

plt.figure(figsize=(8,5))
sns.barplot(x='Day_of_Week', y='Price', data=df)
plt.title("Average Price by Day of Week")
plt.show()

print("--------------------------")

print('Are weekend trips more expensive than other days?')
print(df.groupby('is_weekend')['Price'].mean())

plt.figure(figsize=(6,5))
sns.boxplot(x='is_weekend', y='Price', data=df)
plt.title("Weekend vs Weekday Prices")
plt.show()

print("--------------------------")

print('Does the ticket price vary from month to month?')
print(df.groupby('Month_of_Journey')['Price'].mean())

plt.figure(figsize=(10,5))
sns.lineplot(x='Month_of_Journey', y='Price', data=df, estimator='mean')
plt.gray()
plt.grid()
plt.title("Average Price by Month")
plt.show()

print("--------------------------")

print('Does the peak season lead to higher prices?')
print(df.groupby('is_peak_season')['Price'].mean())

plt.figure(figsize=(6,5))
sns.boxplot(x='is_peak_season', y='Price', data=df)
plt.title("Peak Season vs Non-Peak Season Prices")
plt.show()


print("--------------------------")

print("Does the airline's impact on price vary depending on the number of stops?")
Airline_Price_Total_Stops = pd.pivot_table(df, values='Price', index='Airline', columns='Total_Stops', aggfunc='mean')
print(Airline_Price_Total_Stops)

plt.figure(figsize=(10,6))
sns.heatmap(
    Airline_Price_Total_Stops,annot=True, fmt=".0f", cmap="coolwarm"
)
plt.title("Airline vs Stops (Average Price)")
plt.show()

print("--------------------------")

print('Does the effect of the departure time differ between days of the week?')
Dep_hour_Day_of_Week_onPrice = pd.pivot_table(df, values='Price', index='Dep_hour', columns='Day_of_Week', aggfunc='mean').head()
print(Dep_hour_Day_of_Week_onPrice)

plt.figure(figsize=(10,6))
sns.heatmap(
    Dep_hour_Day_of_Week_onPrice,
    cmap='viridis'
)
plt.title("Departure Hour vs Day of Week (Price)")
plt.show()

print("--------------------------")

print('Does the price of the route change during peak season compared to regular seasons?')
Path_is_Peak_Session_onPrice = pd.pivot_table(df, values='Price', index='Path', columns='is_peak_season', aggfunc='mean')
print(Path_is_Peak_Session_onPrice)

plt.figure(figsize=(10,6))
sns.heatmap(
    Path_is_Peak_Session_onPrice,
    cmap='coolwarm'
)
plt.title("Path Price in Peak vs Non-Peak Season")
plt.show()

print("--------------------------")
print('What is the impact of (airline + number of stops + flight duration) on the price?')
impactOf_Air_Total_Dura_onPrice = df.groupby(['Airline','Total_Stops'])[['Duration_mins','Price']].mean().head()
print(impactOf_Air_Total_Dura_onPrice)

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x='Duration_mins',
    y='Price',
    hue='Total_Stops',
    style='Airline',
    alpha=0.6
)
plt.title("Combined Effect of Airline, Stops & Duration on Price")
plt.grid()
plt.show()

print('=========== Build ML Model ==========')
"""
['Airline', 'Source', 'Destination', 'Total_Stops', 'Price', 'Dep_hour',
'Dep_minute', 'Arrival_hour', 'Arrival_minute', 'Duration_mins',
'Month_of_Journey', 'Days_of_Journey', 'Day_of_Week', 'Quarter',
'is_weekend', 'Path', 'Dep_Session', 'Is_Long_Flight',
'is_peak_season']
"""
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
import numpy as np

# تحديد الأعمدة
categorical_features = ['Airline', 'Source', 'Destination', 'Dep_Session']
numeric_features = ['Total_Stops', 'Dep_hour', 'Arrival_hour', 'Duration_mins',
                    'Month_of_Journey', 'Days_of_Journey', 'Day_of_Week',
                    'is_weekend', 'Is_Long_Flight', 'is_peak_season']

X = df[categorical_features + numeric_features]
y = df['Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ], remainder='passthrough')

# Define Model
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Create Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb)
])

# CV Strategy: KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Proper Metric: MAE (calculated on log scale first for CV)
cv_results = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

print(f"CV MAE Mean: {-cv_results.mean():.4f}")
print(f"CV MAE Std (Stability): {cv_results.std():.4f}")

# Sample Weights calculation
weights = (y_train - y_train.min()) / (y_train.max() - y_train.min()) + 1

# Random Search Params
param_dist = {
    'model__n_estimators': [300 ,500, 750 ,1000 ,1250, 1500],
    'model__learning_rate': [0.001,0.01, 0.05, 0.1],
    'model__max_depth': [3, 6, 9 , 11],
    'model__subsample': [0.7, 0.8, 0.9],
    'model__colsample_bytree': [0.7, 0.8, 0.9],
    'model__gamma': [0,1,3,5]
}

random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=kf, scoring='r2', n_jobs=-1)
random_search.fit(X_train, y_train, model__sample_weight=weights)

print(f"Best Params from RandomSearch: {random_search.best_params_}")

# تضييق النطاق بناءً على نتائج الـ RandomSearch التي حصلت عليها
param_grid = {
    'model__n_estimators': [1400, 1500, 1600],
    'model__max_depth': [10, 11, 12],
    'model__learning_rate': [0.04, 0.05, 0.06],
    'model__gamma': [0.5, 1, 1.5],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8]
}

print("Running Grid Search to fine-tune...")
grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train, model__sample_weight=weights)

print(f"Best Params from GridSearch: {grid_search.best_params_}")

# استخراج أفضل موديل تم الوصول إليه
final_model = grid_search.best_estimator_

# التدريب النهائي باستخدام أوزان العينات لضمان دقة الرحلات النادرة
final_model.fit(X_train, y_train, model__sample_weight=weights)


# 1. التوقع باستخدام أفضل موديل (القيم الناتجة هي Log Price)
y_pred_log = final_model.predict(X_test)

# 2. تحويل القيم من المقياس اللوغاريتمي إلى المقياس الأصلي (السعر الفعلي)
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)

# لضمان عدم وجود قيم سالبة (رغم ندرتها) قبل حساب RMSLE
y_pred_original = np.maximum(y_pred_original, 0)

# 3. حساب المقاييس الاحترافية (Final Evaluation)

# أ. RMSLE: Root Mean Squared Logarithmic Error
# ملاحظة: بما أننا نملك الـ Log بالفعل، يمكن حسابه كـ RMSE للقيم اللوغاريتمية
rmsle = np.sqrt(mean_squared_log_error(y_test_original, y_pred_original))

# ب. MAE: متوسط الخطأ المطلق بالسعر الحقيقي
mae = mean_absolute_error(y_test_original, y_pred_original)

# ج. R2 Score: نسبة شرح التباين
r2 = r2_score(y_test, y_pred_log)

print("-" * 40)
print(f"🚀 FINAL EVALUATION WITH RMSLE")
print("-" * 40)
print(f"RMSLE: {rmsle:.4f}")
print(f"MAE (Actual Price): {mae:.2f} Units")
print(f"R-Squared (Accuracy): {r2:.4%}")
print("-" * 40)


import matplotlib.pyplot as plt
import pandas as pd

# 1. استخراج أهمية المتغيرات من داخل الـ Pipeline
# نحتاج للوصول لأسماء الأعمدة بعد الـ One-Hot Encoding
ohe_columns = list(final_model.named_steps['preprocessor']
                   .named_transformers_['cat']
                   .get_feature_names_out(categorical_features))
all_features = ohe_columns + numeric_features
# دمج
importances = final_model.named_steps['model'].feature_importances_

# 2. تنظيم البيانات في DataFrame
feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

# 3. الرسم البياني
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Top 10 Features Driving Ticket Prices')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

print("============== MLflow LifeCycle =============")

import mlflow.pyfunc
import mlflow.sklearn
import json
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

cv_mae_mean = -cv_results.mean()
cv_mae_std = cv_results.std()
# =========================================================
# 9️⃣ MLflow Wrapper (Production Grade)
# =========================================================

class TicketPriceModelWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def predict(self, context, model_input):

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        missing_cols = set(self.feature_names) - set(model_input.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        model_input = model_input[self.feature_names]

        log_preds = self.model.predict(model_input)
        return np.expm1(log_preds)

# =========================================================
# 🔟 MLflow Lifecycle
# =========================================================

mlflow.set_experiment("Ticket_Price_Prediction_Full_Lifecycle")

with mlflow.start_run(run_name="XGB_Full_Lifecycle") as run:

    run_id = run.info.run_id

    # ---- Params ----
    for k, v in grid_search.best_params_.items():
        mlflow.log_param(k, v)

    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("sample_weight", True)

    # ---- Metrics ----
    mlflow.log_metric("RMSLE", rmsle)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("CV_MAE_Mean", cv_mae_mean)
    mlflow.log_metric("CV_MAE_Std", cv_mae_std)

    # ---- JSON Summary ----
    evaluation_summary = {
        "model": "XGBRegressor",
        "rmsle": rmsle,
        "mae": mae,
        "r2": r2,
        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std": cv_mae_std
    }

    with open("evaluation_summary.json", "w") as f:
        json.dump(evaluation_summary, f, indent=4)

    mlflow.log_artifact("evaluation_summary.json")

    # ---- Feature Importance ----
    feature_importance_df.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    # ---- Signature ----
    input_example = X_train.iloc[:5]
    output_example = np.expm1(final_model.predict(input_example))

    signature = infer_signature(input_example, output_example)

    # ---- PyFunc Model ----
    wrapped_model = TicketPriceModelWrapper(
        final_model,
        feature_names=list(X.columns)
    )

    mlflow.pyfunc.log_model(
        artifact_path="ticket_price_model",
        python_model=wrapped_model,
        input_example=input_example,
        signature=signature,
        registered_model_name="TicketPricePredictor"
    )

    # ---- Registry Control ----
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/ticket_price_model"
    result = mlflow.register_model(
        model_uri=model_uri,
        name="TicketPricePredictor"
    )

    model_version = result.version

    client.transition_model_version_stage(
        name="TicketPricePredictor",
        version=model_version,
        stage="Staging",
        archive_existing_versions=True
    )

    # ---- Production Condition ----
    if r2 >= 0.85 and rmsle <= 0.20 and cv_mae_std <= 0.05:
        client.transition_model_version_stage(
            name="TicketPricePredictor",
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )
        print("🚀 Model promoted to PRODUCTION")
    else:
        print("❌ Model did not meet production criteria")

# =========================================================
# END