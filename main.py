import argparse
import os
import data
import model
import MLflow_LifeCycle
import eda_visualization


def main():
    parser = argparse.ArgumentParser(description="Flight Ticket Price Prediction Pipeline")

    parser.add_argument("--data_path", type=str,
                        default=r"D:\ALL Projects\Ticket\Flight Ticket Price.xlsx",
                        help="Path to the Excel data file")

    parser.add_argument("--n_estimators", type=int, default=1500)
    parser.add_argument("--max_depth", type=int, default=11)
    parser.add_argument("--learning_rate", type=float, default=0.05)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🚀 Flight Ticket Price Prediction System is Starting...")
    print("=" * 60)

    print("\nStep 1: Cleaning and Preprocessing Data...")

    if not os.path.exists(args.data_path):
        print(f"❌ Error: File not found at {args.data_path}")
        return

    df_cleaned = data.run_data_pipeline(args.data_path)
    print(f"✅ Data processed successfully. Shape: {df_cleaned.shape}")

    print("\nStep 1.5: Exploratory Data Analysis (EDA & Visualization)...")
    eda_visualization.run_eda(df_cleaned)

    print("\nStep 2: Training Model with Hyperparameter Tuning...")
    training_results = model.train_model(
        df=df_cleaned,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )
    print("✅ Model training and evaluation completed.")

    print("\nStep 3: Managing MLflow Lifecycle (Logging & Registry)...")

    X_train = training_results['X_train']
    X = training_results['X']

    run_id = MLflow_LifeCycle.run_mlflow_lifecycle(
        training_results=training_results,
        X_train=X_train,
        X=X
    )

    print("\n" + "=" * 60)
    print(f"🎉 Pipeline Execution Finished Successfully!")
    print(f"🆔 Run ID: {run_id}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()