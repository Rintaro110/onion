import data_preprocess as dp
import analyzing_data as ad
import scripts.import_diseasedata as id
import import_meteorologicaldata as im
import output_results as op
import multiprocessing as mp


if __name__ == '__main__':

    # 使用例
    # ファイルパス
    syukaku_data_path = 'resources/desease_data/disease_data_syukaku.xlsx'
    tyozou_data_path = "resources/desease_data/disease_data_tyozou.xlsx"
    fourth_b_data_path = "resources/desease_data/disease_data_4b.xlsx"
    
    month_data_path = 'resources/meteorological_data/nandan_month_12-8.xlsx'  
    syun_data_path = 'resources/meteorological_data/nandan_syun_12-8.xlsx'
    start_year = 1990
    end_year = 2023
    target_varieties = ['ターザン']  # 取得したい品種を指定
    labels = [
        '平均気温(℃)',
        '日最高気温の平均(℃)',
        '日最低気温の平均(℃)',
        '降水量の合計(mm)',
        '日降水量の最大(mm)',
        '平均風速(m/s)',
        '最大風速(m/s)',
        '日照時間(時間)'
    ]
    labels_tyozou = [
        "降水量の合計(mm)_12_下旬",
        "平均風速(m/s)_3_下旬",
        "日照時間(時間)_5_中旬",
        "日最低気温の平均(℃)_1_上旬",
        "日降水量の最大(mm)_3_中旬"
    ]


    labels_fourth_b = {
        "日最低気温の平均(℃)_3_上旬",
        "降水量の合計(mm)_12_上旬",
        "日照時間(時間)_1_中旬",
        "最大風速(m/s)_2_上旬"
    }

    exclde_strings = [
            '_平均',
            '_8',
            '_7'
            '_6',
            '_5',
            '4_下旬',
            ]

    # 病害データをインポート
    # syukaku_desease_data = id.import_desease_data(syukaku_data_path, start_year=start_year, end_year=end_year, target_names=target_varieties, verbose=True)
    tyozou_desease_data = id.import_disease_data(tyozou_data_path, start_year=start_year, end_year=end_year, target_names=target_varieties, verbose=True)
    fourth_b_desease_data = id.import_disease_data(fourth_b_data_path, start_year=start_year, end_year=end_year, target_names=target_varieties, verbose=True)

    # 気象データをインポート
    # meteorological_data_month = tim.import_meteorological_month_data(month_data_path, verbose=True)
    meteorological_data_syun = im.import_meteorological_syun_data(syun_data_path, verbose=True)

    # データ結合
    tyozou_syun_data = dp.preprocess_data(
        tyozou_desease_data, 
        meteorological_data_syun, 
        use_average_only=True, 
        detect_outliers=True, 
        threshold_ratio=0.3,
        target_variable='発病率',
        z_threshold=3,
        exclude_strings=exclde_strings, 
        include_strings=labels,
        apply_correlation_filter=False,
        correlation_threshold=0.8,
        output_file_correlation_pairs="outputs/high_correlation_pairs_syukaku_syun.xlsx",
        output_file_selected_feature="outputs/merged_syukaku_syun_data.csv"
    )

    fourth_b_syun_data = dp.preprocess_data(
        fourth_b_desease_data, 
        meteorological_data_syun, 
        use_average_only=True, 
        detect_outliers=True, 
        threshold_ratio=0.3,
        target_variable='発病率',
        z_threshold=3,
        exclude_strings=exclde_strings, 
        include_strings=labels,
        apply_correlation_filter=False,
        correlation_threshold=0.8,
        output_file_correlation_pairs="outputs/high_correlation_pairs_4b_syun.xlsx",
        output_file_selected_feature="outputs/merged_4b_syun_data.csv"
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
    """ best_models, best_predictors_list, df_final = ad.run_regression_analysis(
        fourth_b_syun_data, 
        method = 'correlation',
        output_file_correlation="outputs/correlation4b_.xlsx"
    ) """

    """ best_models, best_predictors_list, df_final = ad.run_regression_analysis(
        fourth_b_syun_data, 
        method = 'exhaustive',
        top_n=2, 
        max_k=5, 
        cpu_count=mp.cpu_count()-2
    ) """

    best_models, best_predictors_list, df_final = ad.run_regression_analysis(
        fourth_b_syun_data,
        method = 'stepwise',
        labels = labels_fourth_b, 
        top_n=1, 
        max_k=4, 
        n_trials=2
    ) 

    """ best_models, best_predictors_list, df_final = ad.run_regression_analysis(
        syukaku_syun_data, 
        method = 'normal',
        predictors = labels_tyozou, 
    ) """



    # 保存とプロット
    op.save_results_to_excel(best_models, best_predictors_list, filename="regression_results.xlsx", folder="outputs")


    op.plot_multiple_regression(
        best_models,    
        df_final,
        filename_prefix="multiple_regression_plot", 
        folder="outputs", 
        show_years=True)
    
    op.plot_explanatory_vs_target(
        df_final, 
        filename_prefix="explanatory_vs_target", 
        folder="outputs", 
    )