import data_preprocess as dp
import import_weatherdata as wd
import import_diseasedata as dd
import correlation as cor
import pandas as pd
import XGBoost as xgb
import output_results as op
import analyzing_data as ad
import linear_regression as lr

if __name__ == "__main__":
    # 観測日を指定
    wd_file_path = "resources/meteorological_data/1990_2025_sumoto.xlsx"
    dd_file_path = "resources/disease_data/onion_disease_sammary.xlsx"
    
    output_dir = "results_it"
    target_name = "ターザン"
    start_year = 1994
    end_year  = 2023

    period_order = [
        "2月上旬", "2月下旬", "3月上旬", "3月下旬", "4月上旬", "4月下旬", "5月上旬", "5月下旬", "収穫日", "貯蔵調査"
    ]
    target_period = "収穫日"

    # 病害データをインポート
    disease_list= dd.import_disease_data(dd_file_path, target_name, start_year, end_year)
    disease_df = pd.DataFrame(disease_list)

    # データ抽出
    weather_df = wd.extract_meteorological_data(wd_file_path, start_year, end_year)
    # データマージ
    merged_data = dp.preprocess_data(disease_df, weather_df, period_order, use_past_incidence=False)
    # ad.sequential_linear_regression(merged_data, period_order, target_period, max_features=4)
    results = lr.sequential_linear_regression(merged_data, period_order, target_period, max_features=4, n_jobs=8)
    # 結果をエクスポート
    op.save_sequential_regression_results(results, merged_data, output_excel="results_it/sequential_linear_results.xlsx")
    op.plot_sequential_regression_results(merged_data, folder="results_it", filename_prefix="sequential_linear_fit")
