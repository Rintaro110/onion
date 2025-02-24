import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import itertools
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

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
    """
    X = sm.add_constant(df[list(combo)])

    if X.shape[1] > 2:
        # VIFチェック
        valid_vif, high_vif_columns = calculate_vif(X.drop(columns='const'))
        if not valid_vif:
            return None  # VIFが高い場合はこの組み合わせを無視する

    # モデルのフィッティング
    model = sm.OLS(response, X).fit()
    return model

def perfoem_multiple_regression_analysis(df, top_n=1, max_k=None, cpu_count=mp.cpu_count()-2):
    """
    複数の組み合わせに対して並列で回帰分析を行い、最良のtop_n個のモデルを選定する関数。

    Args:
        df (pd.DataFrame): データフレーム
        top_n (int): 最良のモデルの数（デフォルト: 1）
        max_k (int): 最大の説明変数の数（デフォルト: None）
        cpu_count (int): 使用するCPUコアの数（デフォルト: mp.cpu_count()-2）

    Returns:
        list: 最良のモデルのリスト
        list: 最良のモデルの説明変数のリスト
        list: 最良のモデルの説明変数のリスト（定数項を含む）
        pd.DataFrame: 目的変数とメタデータを含む最終結果のデータフレーム
        pd.Series: 目的変数
    """

    print("---------------------------------------------------")
    print("線形重回帰分析を開始します。")

    df = df.reset_index(drop=True)
    meta_data = df[['品種', '年度', '場所']]

    response = df['発病率']
    predictors = [col for col in df.columns if col not in ['発病率', '品種', '年度', '場所']]
    
    # インデックスの整合性を確認（修正）
    if not response.index.equals(df[predictors].index):
        print(f"Response index: {response.index}")
        print(f"Predictors index: {df[predictors].index}")
        raise ValueError("Index mismatch between response and predictors. Ensure they are aligned before proceeding.")

    best_models_adj_r_squared = []
    
    print("max_k:", max_k)
    total_combinations = sum(1 for _ in itertools.combinations(predictors, min(len(predictors), max_k if max_k is not None else len(predictors))))
    print(f"Total combinations to evaluate: {total_combinations}")

    # 全ての組み合わせを取得
    all_combinations = [combo for k in range(1, min(len(predictors), max_k if max_k is not None else len(predictors)) + 1) 
                        for combo in itertools.combinations(predictors, k)]
    
    print(f"Using {cpu_count} CPU cores for parallel processing.")

    # 並列処理でモデルを評価
    with mp.Pool(processes=cpu_count) as pool:
        with tqdm(total=len(all_combinations), desc="Processing combinations") as pbar:
            for result in pool.imap_unordered(partial(process_combination, df=df, response=response), all_combinations):
                if result is not None:
                    model = result
                    if len(best_models_adj_r_squared) < top_n or model.rsquared_adj > best_models_adj_r_squared[-1].rsquared_adj:
                        if len(best_models_adj_r_squared) == top_n:
                            best_models_adj_r_squared.pop()
                        best_models_adj_r_squared.append(model)
                        best_models_adj_r_squared.sort(key=lambda x: x.rsquared_adj, reverse=True)
                pbar.update(1)

    best_models_adj_r_squared = best_models_adj_r_squared[:top_n]
    best_predictors_list = [list(model.model.exog_names[1:]) for model in best_models_adj_r_squared]
    best_X_adj_r_squared = [sm.add_constant(df[predictors]) for predictors in best_predictors_list]
    
    df_results_list = []

    for i, model in enumerate(best_models_adj_r_squared):

        predictions = model.predict(best_X_adj_r_squared[i])
        selected_predictors = best_predictors_list[i]  # このモデルで選ばれた説明変数のリスト
        meta_data_expanded = pd.concat([meta_data] * len(response), ignore_index=True)
        selected_features_df = df[selected_predictors].reset_index(drop=True)

        # モデルの結果をデータフレームに保存
        df_results = pd.DataFrame({
            "モデルID": f"Best_{i+1}",
            "発病率（実測値）": response,
            "発病率（予測値）": predictions,
            "誤差": response - predictions
        })

        # メタデータと選択された説明変数のみを結合
        df_results = pd.concat([meta_data_expanded, df_results, selected_features_df], axis=1)
        df_results_list.append(df_results)

    # すべてのモデルの結果を統合
    df_final = pd.concat(df_results_list, axis=0, ignore_index=True).reset_index(drop=True)


    print("線形重回帰分析が完了しました。")
    print("---------------------------------------------------")

    return best_models_adj_r_squared, best_predictors_list, best_X_adj_r_squared, df_final

