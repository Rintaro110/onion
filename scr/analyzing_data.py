import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import import_weatherdata as wd
import import_diseasedata as dd
import data_preprocess as dp

def train_xgboost_model(df, target_col="incidence", test_size=0.2, random_state=42, output_dir="results_it"):
    """
    Train an XGBoost regression model to predict disease incidence.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with target column and feature columns.
    target_col : str
        The name of the target variable column.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.
    output_dir : str
        Directory to save model and evaluation metrics.

    Returns
    -------
    model : xgb.XGBRegressor
        Trained XGBoost model.
    metrics : dict
        Dictionary containing RMSE and R2 score.
    """

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # Separate features and target
    X = df.drop(columns=[target_col, "date", "brand", "period"])  # drop non-numeric columns
    y = df[target_col]

    # Handle any remaining NaNs
    X = X.fillna(X.mean())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = xgb.XGBRegressor(random_state=random_state, n_estimators=100)
    model.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # squared=False を使わない
    r2 = r2_score(y_test, y_pred)

    metrics = {"rmse": rmse, "r2": r2}

    # Output results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "xgboost_metrics.txt"), "w") as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")

    return model, metrics

if __name__ == "__main__":
    # 観測日を指定
    wd_file_path = "resources/meteorological_data/1990_2025_sumoto.xlsx"
    dd_file_path = "resources/disease_data/onion_disease_sammary.xlsx"
    
    output_dir = "results_it"
    target_name = "ターザン"
    start_year = 1994
    end_year  = 2023

    # 病害データをインポート
    disease_list= dd.import_disease_data(dd_file_path, target_name, start_year, end_year)
    disease_df = pd.DataFrame(disease_list)

    # データ抽出
    weather_df = wd.extract_meteorological_data(wd_file_path, start_year, end_year)
    # データマージ
    merged_data = dp.preprocess_data(disease_df, weather_df)

    # XGBoostモデルのトレーニング
    model, metrics = train_xgboost_model(merged_data, target_col="incidence", output_dir=output_dir)
    print(f"Model trained with RMSE: {metrics['rmse']:.4f}, R2 Score: {metrics['r2']:.4f}")
    # モデルの保存
    model.save_model(os.path.join(output_dir, "xgboost_model.json"))
    print(f"Model saved to {os.path.join(output_dir, 'xgboost_model.json')}")