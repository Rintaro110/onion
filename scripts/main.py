import data_preprocess as dp
import analyzing_data as ad
import import_deseasedata as id
import import_meteorologicaldata as im
import output_results as op
import multiprocessing as mp


if __name__ == '__main__':

    # 使用例
    # ファイルパス
    syukaku_data_path = 'resources/desease_data/disease_data_syukaku.xlsx'
    tyozou_data_path = "resources/desease_data/disease_data_tyozou.xlsx"
    
    month_data_path = 'resources/meteorological_data/nandan_month_12-8.xlsx'  
    syun_data_path = 'resources/meteorological_data/nandan_syun_12-8.xlsx'
    start_year = 1990
    end_year = 2023
    target_varieties = ['ターザン']  # 取得したい品種を指定
    labels = [
        '平均気温(℃)',
        '日最高気温の平均(℃)',
        '日最低気温の平均(℃)',
    ]
    exclde_strings = [
            '_8',
            '_7',
            '_6'
            ]

    # 病害データをインポート
    syukaku_desease_data = id.import_desease_data(syukaku_data_path, start_year=start_year, end_year=end_year, target_names=target_varieties, verbose=True)
    # tyozou_desease_data = tid.import_desease_data(tyozou_data_path, start_year=start_year, end_year=end_year, target_names=target_varieties, verbose=True)
    # 気象データをインポート
    # meteorological_data_month = tim.import_meteorological_month_data(month_data_path, verbose=True)
    meteorological_data_syun = im.import_meteorological_syun_data(syun_data_path, verbose=True)

    # データ結合
    syukaku_syun_data = dp.preprocess_data(
        syukaku_desease_data, 
        meteorological_data_syun, 
        use_average_only=True, 
        detect_outliers=True, 
        z_threshold=3, 
        threshold_ratio=0.3, 
        exclude_strings = exclde_strings,
        include_strings= labels,
        output_file="outputs/merged_syukaku_syun_data.csv"
    )

    """ tyozou_syun_data = tdp.preprocess_data(
        tyozou_desease_data, 
        meteorological_data_syun, 
        use_average_only=True, 
        detect_outliers=True, 
        z_threshold=3, 
        threshold_ratio=0.3, 
        output_file="outputs/merged_tyozou_syun_data.csv"
    ) """


    # 回帰分析
    best_models, best_predictors_list, best_X, final_df = ad.perfoem_multiple_regression_analysis(
        syukaku_syun_data, 
        top_n=2, 
        max_k=3, 
        cpu_count=mp.cpu_count()-2
    )

    # 保存とプロット
    op.save_results_to_excel(best_models, best_predictors_list, filename="regression_results.xlsx", folder="outputs")

    op.plot_multiple_regression(
        best_models,    
        final_df,
        filename_prefix="multiple_regression_plot", 
        folder="outputs", 
        show_years=True)
    
    op.plot_explanatory_vs_target(
        final_df, 
        filename_prefix="explanatory_vs_target", 
        folder="outputs", 
    )