import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import json
import joblib
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


class TicketPriceModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_file"])

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self.feature_names)

        model_input = model_input[self.feature_names]

        log_preds = self.model.predict(model_input)
        return np.expm1(log_preds)


def run_mlflow_lifecycle(training_results, X_train, X):
    print("\n" + "=" * 20 + " MLflow LifeCycle " + "=" * 20)

    final_model = training_results['model']
    rmsle = training_results['rmsle']
    mae = training_results['mae']
    r2 = training_results['r2']
    best_params = training_results['params']
    fi_df = training_results.get('feature_importance_df')

    mlflow.set_experiment("Ticket_Price_Prediction_Full_Lifecycle")


    with mlflow.start_run(run_name="XGB_Full_Lifecycle") as run:
        run_id = run.info.run_id

        search_configs = {
            "random_search_space": training_results.get('search_space_random'),
            "grid_search_space": training_results.get('search_space_grid')
        }
        with open("search_configs.json", "w") as f:
            json.dump(search_configs, f, indent=4)
        mlflow.log_artifact("search_configs.json")

        mlflow.log_param("rs_n_estimators_list",str(training_results.get('search_space_random').get('model__n_estimators')))
        mlflow.log_param("rs_max_depth_list", str(training_results.get('search_space_random').get('model__max_depth')))
        mlflow.log_param("gs_gamma_options", str(training_results.get('search_space_grid').get('model__gamma')))

        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("cv_folds", 5)

        mlflow.log_param("dataset_shape", str(X.shape))
        mlflow.log_param("test_size_ratio", 0.2)
        mlflow.log_param("features_count", len(X.columns))


        mlflow.log_metrics({
            "RMSLE": float(rmsle),
            "MAE": float(mae),
            "R2_Score": float(r2),
            "cv_mae_mean": float(training_results.get('cv_mae_mean', 0)),
            "cv_mae_std": float(training_results.get('cv_mae_std', 0))
        })

        if fi_df is not None:
            fi_df.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

        evaluation_summary = {"model": "XGBRegressor", "rmsle": rmsle, "mae": mae, "r2": r2}
        with open("evaluation_summary.json", "w") as f:
            json.dump(evaluation_summary, f, indent=4)
        mlflow.log_artifact("evaluation_summary.json")

        model_filename = "xgb_model.pkl"
        joblib.dump(final_model, model_filename)

        artifacts = {
            "model_file": model_filename
        }

        input_example = X_train.iloc[:5]
        output_example = pd.DataFrame(np.expm1(final_model.predict(input_example)), columns=["price_prediction"])
        signature = infer_signature(input_example, output_example)

        wrapped_model = TicketPriceModelWrapper(feature_names=list(X.columns))

        mlflow.pyfunc.log_model(
            artifact_path="ticket_price_model",
            python_model=wrapped_model,
            artifacts=artifacts,
            input_example=input_example,
            signature=signature,
            registered_model_name="TicketPricePredictor"
        )

        client = MlflowClient()
        model_name = "TicketPricePredictor"
        model_uri = f"runs:/{run_id}/ticket_price_model"

        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        model_version = result.version

        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True
        )

        if r2 >= 0.85 and rmsle <= 0.20:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Production",
                archive_existing_versions=True
            )
            status = "Production 🚀"
        else:
            status = "Staging 🛑 (Quality Gate Failed)"

        print(f"📦 Model Version: {model_version}")
        print(f"🏷️ Final Status: {status}")
        print(f"🆔 Run ID: {run_id}")
        print("=" * 50)

        return run_id