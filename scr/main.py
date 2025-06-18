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
    ]
    "storage_survey"
    target_period = "harvest"
    target_col = "log_incidence"

    # 病害データをインポート
    disease_list= dd.import_disease_data(dd_file_path, target_name, start_year, end_year)
    disease_df = pd.DataFrame(disease_list)

    # データ抽出
    weather_df = wd.extract_meteorological_data(wd_file_path, start_year, end_year, lang = "en")
    # 出力ディレクトリを作成
    output_dir="results_linear_regression_1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 出力ディレクトリを指定

    # データマージ for disease analysis
    """ merged_df = dp.preprocess_data(
        disease_df, 
        weather_df, 
        period_order, 
        use_past_incidence=True, 
        remove_outliers=False,
        output_dir=output_dir,
        test_years=None
    ) """

    # データマージ for modeling
    train_df, test_df = dp.preprocess_data(
        disease_df, 
        weather_df, 
        period_order, 
        use_past_incidence=False, 
        remove_outliers=False,
        output_dir=output_dir,
        test_years= [2022, 2023, 2024]  # モデル評価用のテストデータ年を指定
    )
   

    """ dd.plot_disease_trends_by_year(
        merged_df, 
        target_name="ターザン", 
        output_dir=output_dir, 
        show=True, 
        log=True, 
        log_col="log_incidence"
    ) """

    # 相関分析
    # cor.analyze_and_export_correlation(merged_df, target_type="cor1", target_col=target_col,  output_path=output_dir)
    # cor.analyze_and_export_correlation(merged_df, target_type="cor2", target_col=target_col,  output_path=output_dir)
    
    # modeling
    # 線形回帰モデルの実行
    train_result_df, test_result_df, train_all, test_all = lr.sequential_regression(train_df, test_df, target_col="log_incidence", periods_order=period_order, max_features=4, n_jobs=8)
    # 結果をエクスポート
    op.save_results(train_result_df, test_result_df, train_all, test_all, output_dir=output_dir) 

    # --------------------------------------
    # xgboostモデルの実行
    """ train_result_df, test_result_df, train_all, test_all = xgb.sequential_xgboost(train_df, test_df, target_col="log_incidence", periods_order=period_order)
    # 結果をエクスポート
    op.save_results(train_result_df, test_result_df, train_all, test_all, output_dir=output_dir) """  
