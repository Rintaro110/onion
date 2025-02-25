import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import itertools
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import random

import output_results as op

def calculate_vif(X):
    """VIFを計算し、10を超えるVIFがあるかどうかを返し、VIFが高い列のリストを返す。"""
    
    # 分散がほぼ0のカラムを除去
    X = X.loc[:, X.var() > 1e-10]

    # Xが空、または列が2つ未満の場合、VIF計算をスキップ
    if X.empty or len(X.columns) < 2:
        return True, []  # 問題なしと判断

    # VIFスコアを計算
    vif_scores = []
    for i, column in enumerate(X.columns):
        try:
            # 説明変数Xから対象の列を除外し、定数項を追加
            X_excluded = sm.add_constant(X.iloc[:, np.arange(X.shape[1]) != i])

            # R^2を計算
            model = sm.OLS(X.iloc[:, i], X_excluded).fit()
            r_squared_i = model.rsquared

            # R^2が1に極端に近い場合、多重共線性の疑いがあるため np.inf を設定
            if r_squared_i > 0.999:
                vif = np.inf
            else:
                vif = 1. / (1. - r_squared_i)
                
            vif_scores.append((column, vif))

        except Exception as e:
            # エラー発生時は np.inf を割り当てる
            vif_scores.append((column, np.inf))

    # VIF > 10 のカラムをリスト化
    high_vif_columns = [col for col, vif in vif_scores if vif > 10]

    return len(high_vif_columns) == 0, high_vif_columns  # すべてのVIFが10未満ならTrue

def process_combination(combo, df, response):
    """
    各組み合わせに対して回帰分析を行い、VIFのチェックを行う。

    Args:
        combo (tuple): 使用する説明変数の組み合わせ
        df (pd.DataFrame): 説明変数を含むデータフレーム
        response (pd.Series): 目的変数

    Returns:
        tuple: (回帰モデル, 説明変数リスト) または (None, None) (VIFチェックで失敗時)
    """
    X = sm.add_constant(df[list(combo)])

    # インデックスをリセットして整合性を保つ
    X = X.reset_index(drop=True)
    response = response.reset_index(drop=True)

    if X.shape[1] > 2:
        # VIFチェック
        valid_vif, high_vif_columns = calculate_vif(X.drop(columns='const'))
        if not valid_vif:
            return None, None  # VIFが高い場合はこの組み合わせを無視する

    # モデルのフィッティング
    model = sm.OLS(response, X).fit()

    return model, list(combo)  # 説明変数リストをタプルで返す

def prepare_data(df):
    """
    データの前処理を行い、目的変数・説明変数・メタデータを取得する。
    Args:
        df (pd.DataFrame): データフレーム

    Returns:
        tuple: (meta_data, response, predictors)
    """
    df = df.reset_index(drop=True)
    meta_data_columns = ['品種', '年度', '場所']

    if not all(col in df.columns for col in meta_data_columns):
        raise ValueError("データフレームに '品種', '年度', '場所' のカラムが必要です。")

    if '発病率' not in df.columns:
        raise ValueError("データフレームに '発病率' (目的変数) が含まれていません。")

    meta_data = df[meta_data_columns]
    response = df['発病率']
    predictors = [col for col in df.columns if col not in ['発病率'] + meta_data_columns]

    if not predictors:
        raise ValueError("説明変数が存在しません。")

    return meta_data, response, predictors

def evaluate_model(response, df, predictors):
    """
    指定された説明変数で回帰モデルを構築し、評価する。
    
    Args:
        response (pd.Series): 目的変数
        df (pd.DataFrame): 入力データフレーム
        predictors (list): 説明変数のリスト
    
    Returns:
        tuple: (修正決定係数, モデルオブジェクト, 使用した説明変数リスト)
    """
    X = sm.add_constant(df[predictors])
    X = X.reset_index(drop=True)
    response = response.reset_index(drop=True)
    model = sm.OLS(response, X).fit()
    
    return model.rsquared_adj, model, predictors

def generate_results(best_models, best_predictors_list, df, meta_data, response):
    """
    最適なモデルの出力データフレームを作成する。
    
    Args:
        best_models (list): statsmodels のモデルオブジェクトのリスト
        best_predictors_list (list): 各モデルの説明変数のリスト
        df (pd.DataFrame): 入力データフレーム
        meta_data (pd.DataFrame): メタデータ
        response (pd.Series): 目的変数

    Returns:
        pd.DataFrame: モデルの結果データフレーム
    """
    df_results_list = []

    for i, model in enumerate(best_models):
        best_X = sm.add_constant(df[best_predictors_list[i]])
        predictions = model.predict(best_X)

        df_results = pd.DataFrame({
            "モデルID": f"Best_{i+1}",
            "発病率（実測値）": response,
            "発病率（予測値）": predictions,
            "誤差": response - predictions
        })

        df_results = pd.concat([meta_data, df_results, df[best_predictors_list[i]]], axis=1)
        df_results_list.append(df_results)

    return pd.concat(df_results_list, axis=0, ignore_index=True)

def perform_multiple_regression(df, selected_features):
    """
    指定した説明変数で重回帰分析を行う関数。

    Args:
        df (pd.DataFrame): 入力データフレーム
        predictors (list): 使用する説明変数のリスト

    Returns:
        list: statsmodels のモデルオブジェクトのリスト
        list: 各モデルの説明変数のリスト
        pd.DataFrame: 回帰分析の結果を含むデータフレーム
    """

    print("---------------------------------------------------")
    print("指定された説明変数で線形重回帰分析を実行します。")

    # データの前処理
    meta_data, response, predictors = prepare_data(df)

    if selected_features is None:
        raise ValueError("説明変数が指定されていません。")
    print(f"使用する説明変数: {selected_features}")

    
    # データに存在する説明変数のみを取得
    valid_features = [col for col in selected_features if col in predictors]
    missing_features = [col for col in selected_features if col not in predictors]

    if missing_features:
        raise ValueError(f"指定された説明変数がデータに含まれていません: {missing_features}")
    
    valid_X = df[valid_features].reset_index(drop=True)
    response = response.reset_index(drop=True)

    adj_r_squared, model, selected_predictors = evaluate_model(response, valid_X, valid_features)


    # top_n のモデルを1つだけ保持
    best_models_adj_r_squared = [(adj_r_squared, model, selected_predictors)]
    best_models = [model for _, model, _ in best_models_adj_r_squared]
    best_predictors_list = [predictors for _, _, predictors in best_models_adj_r_squared]

    # 結果のデータフレームを作成
    df_final = generate_results(best_models, best_predictors_list, df, meta_data, response)

    print("線形重回帰分析が完了しました。")
    print(f"最良のモデルの決定係数 (R²_adj): {round(adj_r_squared, 4)}")
    print("---------------------------------------------------")

    return best_models, best_predictors_list, df_final

def perform_stepwise_multiple_regression(df, labels=None, top_n=1, max_k=None, n_trials=10, random_seed=42):
    """
    逐次選択法による回帰分析を実施。
    """

    print("---------------------------------------------------")
    print("逐次選択法による線形重回帰分析を開始します。")

    meta_data, response, predictors = prepare_data(df)

    if labels is None:
        labels = []
        print("必須説明変数 is None")
    
    # 最大説明変数数の設定
    if max_k is None:
        max_k = len(predictors)
    
    print(f"必須変数 (labels): {labels}")
    print(f"最大 {max_k} 個の説明変数を選択します。")
    print(f"{n_trials} 回ランダムな順番で試行し、最良の {top_n} モデルを選択します。")

    best_models_adj_r_squared = []
 
    for trial in range(n_trials):
        print(f"\n試行 {trial+1}/{n_trials}")

        current_predictors = labels.copy()
        random.seed(random_seed + trial)
        shuffled_predictors = random.sample(predictors, len(predictors))

        pbar = tqdm(total=max_k, desc=f"逐次選択 (試行 {trial+1})")

        while len(current_predictors) - len(labels) < max_k:
            remaining_predictors = list(set(shuffled_predictors) - set(current_predictors))

            if not remaining_predictors:
                print(f"試行 {trial+1}: 追加可能な変数がなくなったため終了します。")
                pbar.update(1)
                break

            best_model = None
            best_adj_r_squared = -float('inf')
            best_new_predictor = None

            for predictor in remaining_predictors:
                if predictor not in df.columns:
                    print(f"Warning: {predictor} is not in df.columns, skipping...")
                    continue

                test_predictors = current_predictors + [predictor]
                X = sm.add_constant(df[test_predictors])

                # VIFチェックを実施
                if X.shape[1] > 2:
                    valid_vif, high_vif_columns = calculate_vif(X.drop(columns='const'))
                    if not valid_vif:
                        # print(f"VIFチェック: {high_vif_columns} が高VIFのためスキップ")
                        continue  

                # モデルの作成・評価
                adj_r_squared, model, _ = evaluate_model(response, df, test_predictors)

                if adj_r_squared > best_adj_r_squared:
                    best_adj_r_squared = adj_r_squared
                    best_model = model
                    best_new_predictor = predictor

            if best_new_predictor is None:
                print(f"試行 {trial+1}: 追加可能な変数がなくなったため終了します。")
                pbar.update(1)
                break

            # 説明変数の更新
            current_predictors.append(best_new_predictor)

            # top_n のモデルを管理
            if len(best_models_adj_r_squared) < top_n or best_model.rsquared_adj > best_models_adj_r_squared[-1][0]:
                if len(best_models_adj_r_squared) == top_n:
                    best_models_adj_r_squared.pop()
                best_models_adj_r_squared.append((best_model.rsquared_adj, best_model, list(current_predictors)))
                best_models_adj_r_squared.sort(reverse=True, key=lambda x: x[0])

            pbar.update(1)

        pbar.close()

    best_models = [model for _, model, _ in best_models_adj_r_squared]
    best_predictors_list = [predictors for _, _, predictors in best_models_adj_r_squared]

    df_final = generate_results(best_models, best_predictors_list, df, meta_data, response)

    print("線形重回帰分析が完了しました。")
    print(f"最良のモデルの決定係数 (R²_adj): {[round(x[0], 4) for x in best_models_adj_r_squared]}")
    print("---------------------------------------------------")

    return best_models, best_predictors_list, df_final

def perfoem_multiple_regression_summation(df, top_n=1, max_k=None, cpu_count=mp.cpu_count()-2):
    """
    総当たり法を使用して最適な回帰モデルを選定。

    Args:
        df (pd.DataFrame): データフレーム
        top_n (int): 最良のモデルの数（デフォルト: 1）
        max_k (int, optional): 最大の説明変数の数（デフォルト: None → すべての変数が選択可能）
        cpu_count (int): 使用するCPUコアの数（デフォルト: mp.cpu_count()-2）

    Returns:
        list: statsmodelsのモデルオブジェクトのリスト
        list: 各モデルの説明変数のリスト
        pd.DataFrame: 回帰分析の結果を含むデータフレーム
    """

    print("---------------------------------------------------")
    print("線形重回帰分析(総当たり法)を開始します。")

    meta_data, response, predictors = prepare_data(df)

    if max_k is None:
        max_k = len(predictors)

    all_combinations = [
        combo for k in range(1, min(len(predictors), max_k) + 1) 
        for combo in itertools.combinations(predictors, k)
    ]

    print(f"最大 {max_k} 個の説明変数を使用して回帰分析を行います。")
    print(f"Using {cpu_count} CPU cores for parallel processing.")
    print(f"Total number of combinations: {len(all_combinations)}")

    best_models_adj_r_squared = []

    # 並列処理でモデルを評価
    with mp.Pool(processes=cpu_count) as pool:
        with tqdm(total=len(all_combinations), desc="Processing combinations") as pbar:
            for result in pool.imap_unordered(partial(process_combination, df=df, response=response), all_combinations):
                if result is None or result[0] is None:
                    continue  # None が含まれる場合はスキップ

                model, selected_predictors = result  # ここで安全にアンパック
                adj_r_squared, model, selected_predictors = evaluate_model(response, df, selected_predictors)  # モデル評価

                if len(best_models_adj_r_squared) < top_n or adj_r_squared > best_models_adj_r_squared[-1][0]:
                    if len(best_models_adj_r_squared) == top_n:
                        best_models_adj_r_squared.pop()
                    best_models_adj_r_squared.append((adj_r_squared, model, list(selected_predictors)))
                    best_models_adj_r_squared.sort(reverse=True, key=lambda x: x[0])  # R²_adj の降順でソート

                pbar.update(1)


    best_models = [model for _, model, _ in best_models_adj_r_squared]
    best_predictors_list = [predictors for _, _, predictors in best_models_adj_r_squared]

    df_final = generate_results(best_models, best_predictors_list, df, meta_data, response)

    print("線形重回帰分析が完了しました。")
    print(f"最良のモデルの決定係数 (R²_adj): {[round(x[0], 4) for x in best_models_adj_r_squared]}")
    print("---------------------------------------------------")

    return best_models, best_predictors_list, df_final

def perform_correlation_analysis_and_save_to_excel(df, filename="outputs/correlation_analysis.xlsx"):
    """
    発病率と気象データの相関係数を計算し、エクセルファイルに書き出す（相関係数の値で細かく色分け）。

    :param df: データフレーム
    :param filename: 保存するエクセルファイルの名前
    """

    print("---------------------------------------------------")
    print("線形単回帰分析(相関分析)を開始します。")

    # データの前処理
    meta_data, response, predictors = prepare_data(df)

    if not predictors:
        raise ValueError("説明変数が存在しません。")

    # 発病率と各気象データの相関係数を計算
    correlation_results = {}
    for predictor in predictors:
        correlation = df[[response.name, predictor]].corr().iloc[0, 1]  # 発病率との相関
        correlation_results[predictor] = correlation

    # 結果をデータフレームに変換
    correlation_df = pd.DataFrame(list(correlation_results.items()), columns=['気象情報', '相関係数'])

    # 相関係数で降順にソート
    correlation_df = correlation_df.sort_values(by='相関係数', ascending=False)

    # 結果をエクセルファイルに保存（色付き）
    op.save_correlation_results_with_colors(correlation_df, filename)
    print(f"相関分析結果を {filename} に保存しました。")
    print("---------------------------------------------------")

    return None, None, None


def run_regression_analysis(df, 
                            method="stepwise", 
                            labels=None, 
                            predictors = None, 
                            top_n=1, 
                            max_k=None, 
                            n_trials=10, 
                            random_seed=42, 
                            cpu_count=mp.cpu_count()-2,
                            output_file_correlation = "outputs/correlation_analysis.xlsx"):
                            
    """
    逐次選択法または総当たり法で最適な回帰モデルを選定。

    Args:
        df (pd.DataFrame): データフレーム
        method (str): "stepwise" (逐次選択法) または "exhaustive" (総当たり法)
        labels (list, optional): 逐次選択法で常に含める説明変数のリスト（デフォルト: None）
        predictors (list, optional): 使用する説明変数のリスト（デフォルト: None）
        top_n (int): 最良のモデルの数（デフォルト: 1）
        max_k (int, optional): 最大の説明変数の数（デフォルト: None → すべての変数が選択可能）
        n_trials (int): 逐次選択法で試行する回数（デフォルト: 10）
        random_seed (int): ランダムシード値（デフォルト: 42）
        cpu_count (int): 総当たり法で使用するCPUコアの数（デフォルト: mp.cpu_count()-2）
        output_file_correlation (str): 相関分析の結果を保存するエクセルファイルのパス（デフォルト: "outputs/correlation_analysis.xlsx")

    Returns:
        list: statsmodelsのモデルオブジェクトのリスト
        list: 各モデルの説明変数のリスト
        pd.DataFrame: 回帰分析の結果を含むデータフレーム
    """

    if method == "stepwise":
        return perform_stepwise_multiple_regression(df, labels=labels, top_n=top_n, max_k=max_k, n_trials=n_trials, random_seed=random_seed)
    elif method == "exhaustive":
        return perfoem_multiple_regression_summation(df, top_n=top_n, max_k=max_k, cpu_count=cpu_count)
    elif method == "normal":
        return perform_multiple_regression(df, predictors)
    elif method == "correlation":
        return perform_correlation_analysis_and_save_to_excel(df, filename=output_file_correlation)
    else:
        raise ValueError("Invalid method. Choose 'stepwise' or 'exhaustive'.")

