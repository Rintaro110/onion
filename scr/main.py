import data_preprocess as dp
import import_weatherdata as wd
import import_diseasedata as dd
import correlation as cor
import os
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
    end_year  = 2024

    period_order = [
        "early_February",
        "late_February",
        "early_March",
        "late_March",
        "early_April",
        "late_April",     
        "early_May",
        "late_May",
        "harvest",
        "storage_survey"
    ]
    target_period = "harvest"
    target_col = "log_incidence"

    # 病害データをインポート
    disease_list= dd.import_disease_data(dd_file_path, target_name, start_year, end_year)
    disease_df = pd.DataFrame(disease_list)

    # データ抽出
    weather_df = wd.extract_meteorological_data(wd_file_path, start_year, end_year, lang = "en")
    # 出力ディレクトリを作成
    output_dir="results_it"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 出力ディレクトリを指定

    # データマージ
    merged_df = dp.preprocess_data(
        disease_df, 
        weather_df, 
        period_order, 
        use_past_incidence=True, 
        output_dir=output_dir, 
        test_years=None
    )
   
    cor.analyze_and_export_correlation(merged_df, target_type="cor1", target_col=target_col,  output_path="results_it/correlation_output.xlsx")
    # cor.analyze_and_export_correlation(merged_df, target_type="cor2", target_col=target_col,  output_path="results_it/correlation_output.xlsx")
    """ # ad.sequential_linear_regression(merged_data, period_order, target_period, max_features=4)
    # train_result_df, test_result_df, train_all, test_all = lr.sequential_regression(train_df, test_df, target_col="incidence", periods_order=period_order, max_features=2, n_jobs=6)
    train_result_df, test_result_df, train_all, test_all = xgb.sequential_xgboost(train_df, test_df, target_col="incidence", periods_order=period_order)
    # 結果をエクスポート
    df_final = op.save_sequential_results(train_result_df, test_result_df, train_all, test_all, output_excel="results_it/sequential_xgboost_results.xlsx")

    # 全体評価
    summary = op.evaluate_sequential_model(train_result_df, test_result_df)
    op.plot_sequential_model_results(train_result_df, test_result_df, output_dir="results_it")
    """