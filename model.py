import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
import logging
logger = logging.getLogger(__name__)

def train_model(df, n_estimators, max_depth, learning_rate):

    logger.info('=========== Build ML Model ==========')
    """
    ['Airline', 'Source', 'Destination', 'Total_Stops', 'Price', 'Dep_hour',
    'Dep_minute', 'Arrival_hour', 'Arrival_minute', 'Duration_mins',
    'Month_of_Journey', 'Days_of_Journey', 'Day_of_Week', 'Quarter',
    'is_weekend', 'Path', 'Dep_Session', 'Is_Long_Flight',
    'is_peak_season']
    """

    categorical_features = ['Airline', 'Source', 'Destination', 'Dep_Session']
    numeric_features = ['Total_Stops', 'Dep_hour', 'Arrival_hour', 'Duration_mins',
                        'Month_of_Journey', 'Days_of_Journey', 'Day_of_Week',
                        'is_weekend', 'Is_Long_Flight', 'is_peak_season']

    X = df[categorical_features + numeric_features]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    preprocessor = ColumnTransformer(
        transformers=[
            ('cat',OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ], remainder='passthrough')

    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb)
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

    logger.info(f"CV MAE Mean: {-cv_results.mean():.4f}")
    logger.info(f"CV MAE Std (Stability): {cv_results.std():.4f}")

    weights = (y_train - y_train.min()) / (y_train.max() - y_train.min()) + 1

    param_dist = {
        'model__n_estimators': [500, 1000, n_estimators],
        'model__learning_rate': [0.01, learning_rate, 0.1],
        'model__max_depth': [max_depth, 6, 9],
        'model__subsample': [0.8, 0.9],
        'model__colsample_bytree': [0.8, 0.9],
        'model__gamma': [0, 1, 5]
    }
    logger.info(f"Parametrs RS:{param_dist}")

    random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=kf, scoring='r2', n_jobs=-1)
    random_search.fit(X_train, y_train, model__sample_weight=weights)

    logger.info(f"Best Params from RandomSearch: {random_search.best_params_}")

    param_grid = {
        'model__n_estimators': [n_estimators],
        'model__max_depth': [max_depth],
        'model__learning_rate': [learning_rate],
        'model__gamma': [0.5, 1, 1.5],
        'model__subsample': [0.8],
        'model__colsample_bytree': [0.8]
    }
    logger.info(f"Parametrs GS:{param_grid}")

    logger.info("Running Grid Search to fine-tune...")
    #
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train, model__sample_weight=weights)

    logger.info(f"Best Params from GridSearch: {grid_search.best_params_}")

    final_model = grid_search.best_estimator_

    final_model.fit(X_train, y_train, model__sample_weight=weights)

    y_pred_log = final_model.predict(X_test)

    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred_log)

    y_pred_original = np.maximum(y_pred_original, 0)

    rmsle = np.sqrt(mean_squared_log_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test, y_pred_log)

    logger.info("-" * 40)
    logger.info(f"🚀 FINAL EVALUATION WITH RMSLE")
    logger.info("-" * 40)
    logger.info(f"RMSLE: {rmsle:.4f}")
    logger.info(f"MAE (Actual Price): {mae:.2f} Units")
    logger.info(f"R-Squared (Accuracy): {r2:.4%}")
    logger.info("-" * 40)

    ohe_columns = list(final_model.named_steps['preprocessor']
                       .named_transformers_['cat']
                       .get_feature_names_out(categorical_features))
    all_features = ohe_columns + numeric_features
    importances = final_model.named_steps['model'].feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Top 10 Features Driving Ticket Prices')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()

    return {
        "model": final_model,
        "rmsle": rmsle,
        "mae": mae,
        "r2": r2,
        "cv_mae_mean": -cv_results.mean(),
        "cv_mae_std": cv_results.std(),
        "params": grid_search.best_params_,
        "search_space_random": param_dist,
        "search_space_grid": param_grid,
        "feature_importance_df": feature_importance_df,
        "X_train": X_train,
        "X": X
    }









