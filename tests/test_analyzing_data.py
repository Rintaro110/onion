import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import itertools
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import random

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

    if X.shape[1] > 2:
        # VIFチェック
        valid_vif, high_vif_columns = calculate_vif(X.drop(columns='const'))
        if not valid_vif:
            return None, None  # VIFが高い場合はこの組み合わせを無視する

    # モデルのフィッティング
    model = sm.OLS(response, X).fit()

    return model, list(combo)  # 説明変数リストをタプルで返す


def perfoem_multiple_regression_summation(df, top_n=1, max_k=None, cpu_count=mp.cpu_count()-2):
    """
    複数の組み合わせに対して並列で回帰分析を行い、最良の top_n 個のモデルを選定する関数。

    Args:
        df (pd.DataFrame): データフレーム
        top_n (int): 最良のモデルの数（デフォルト: 1）
        max_k (int): 最大の説明変数の数（デフォルト: None）
        cpu_count (int): 使用するCPUコアの数（デフォルト: mp.cpu_count()-2）

    Returns:
        list: statsmodelsのモデルオブジェクトのリスト
        list: 各モデルの説明変数のリスト
        pd.DataFrame: 回帰分析の結果を含むデータフレーム（Fitted vs Actual プロット用）
    """

    print("---------------------------------------------------")
    print("線形重回帰分析(総当たり法)を開始します。")

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
        print("説明変数がありません。分析を実行できません。")
        return None, None, None

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
                adj_r_squared = model.rsquared_adj  # ここで NoneType になることはなくなる

                if len(best_models_adj_r_squared) < top_n or adj_r_squared > best_models_adj_r_squared[-1][0]:
                    if len(best_models_adj_r_squared) == top_n:
                        best_models_adj_r_squared.pop()
                    best_models_adj_r_squared.append((adj_r_squared, model, list(selected_predictors)))
                    best_models_adj_r_squared.sort(reverse=True, key=lambda x: x[0])  # R²_adj の降順でソート

                pbar.update(1)

    # top_n のモデルを取得
    best_models = [model for _, model, _ in best_models_adj_r_squared]
    best_predictors_list = [predictors for _, _, predictors in best_models_adj_r_squared]

    # 結果の整理
    df_results_list = []

    for i, (_, model, selected_predictors) in enumerate(best_models_adj_r_squared):

        best_X = sm.add_constant(df[selected_predictors])
        predictions = model.predict(best_X)

        # モデルの結果をデータフレームに保存
        df_results = pd.DataFrame({
            "モデルID": f"Best_{i+1}",
            "発病率（実測値）": response,
            "発病率（予測値）": predictions,
            "誤差": response - predictions
        })

        # メタデータと選択された説明変数のみを結合
        df_results = pd.concat([meta_data, df_results, df[selected_predictors]], axis=1)
        df_results_list.append(df_results)

    # すべてのモデルの結果を統合
    df_final = pd.concat(df_results_list, axis=0, ignore_index=True).reset_index(drop=True)

    print("線形重回帰分析が完了しました。")
    print(f"最良のモデルの決定係数 (R²_adj): {[round(x[0], 4) for x in best_models_adj_r_squared]}")
    print("---------------------------------------------------")

    return best_models, best_predictors_list, df_final


def perform_stepwise_multiple_regression(df, labels=None, top_n=1, max_k=None, n_trials=10, random_seed=42):
    """
    逐次選択法を使用して最適な重回帰モデルを構築し、最良の top_n 個のモデルを選定する。
    - 追加する説明変数の順番をランダムにし、複数回試行して最良の結果を出力
    - 必須変数 (labels) は常にモデルに含める

    Args:
        df (pd.DataFrame): データフレーム
        labels (list, optional): 逐次選択法で常に含める説明変数のリスト（デフォルト: None）
        top_n (int): 最良のモデルの数（デフォルト: 1）
        max_k (int, optional): 最大の説明変数の数（デフォルト: None → すべての変数が選択可能）
        n_trials (int): 変数の追加順をランダムにして試行する回数（デフォルト: 10）
        random_seed (int): ランダムシード値（デフォルト: 42）

    Returns:
        list: statsmodelsのモデルオブジェクトのリスト
        list: 各モデルの説明変数のリスト
        pd.DataFrame: 回帰分析の結果を含むデータフレーム（Fitted vs Actual プロット用）
    """

    print("---------------------------------------------------")
    print("逐次選択法による線形重回帰分析を開始します。")

    df = df.reset_index(drop=True)

    # meta_data の処理 
    meta_data_columns = ['品種', '年度', '場所']
    if not all(col in df.columns for col in meta_data_columns):
        raise ValueError("データフレームに '品種', '年度', '場所' のカラムが必要です。")
    # 目的変数
    if '発病率' not in df.columns:
        raise ValueError("データフレームに '発病率' (目的変数) が含まれていません。")
    
    meta_data = df[meta_data_columns]
    response = df['発病率']
    predictors = [col for col in df.columns if col not in ['発病率'] + meta_data_columns]

    # labels: 逐次選択法で必ず含める変数
    if labels is None:
        labels = []

    if len(predictors) == 0:
        print("説明変数がありません。分析を実行できません。")
        return None, None, None

    if max_k is None:
        max_k = len(predictors)

    print(f"必須変数 (labels): {labels}")
    print(f"最大 {max_k} 個の説明変数を選択します。")
    print(f"{n_trials} 回ランダムな順番で試行し、最良の {top_n} モデルを選択します。")

    best_models_adj_r_squared = []

    for trial in range(n_trials):
        print(f"\n試行 {trial+1}/{n_trials}")

        current_predictors = labels.copy()
        pbar = tqdm(total=max_k, desc=f"逐次選択 (試行 {trial+1})")

        random.seed(random_seed + trial)
        shuffled_predictors = random.sample(predictors, len(predictors))

        while len(current_predictors) - len(labels) < max_k:
            remaining_predictors = list(set(shuffled_predictors) - set(current_predictors))

            if not remaining_predictors:  # 追加可能な変数がなくなったら終了
                print(f"試行 {trial+1}: 追加可能な変数がなくなったため終了します。")
                pbar.update(1)  # 進捗を強制的に更新
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

                # VIF チェックを実施
                if X.shape[1] > 2:
                    valid_vif, high_vif_columns = calculate_vif(X.drop(columns='const'))
                    if not valid_vif:
                        continue  # VIFが高い場合、この組み合わせはスキップ

                model = sm.OLS(response, X).fit()

                if model.rsquared_adj > best_adj_r_squared:
                    best_adj_r_squared = model.rsquared_adj
                    best_model = model
                    best_new_predictor = predictor

            if best_new_predictor is None:
                print(f"試行 {trial+1}: 追加可能な変数がなくなったため終了します。")
                pbar.update(1)  # 進捗を強制的に更新
                break

            current_predictors.append(best_new_predictor)

            # top_n のモデルを管理
            if len(best_models_adj_r_squared) < top_n or best_model.rsquared_adj > best_models_adj_r_squared[-1][0]:
                if len(best_models_adj_r_squared) == top_n:
                    best_models_adj_r_squared.pop()
                best_models_adj_r_squared.append((best_model.rsquared_adj, best_model, list(current_predictors)))
                best_models_adj_r_squared.sort(reverse=True, key=lambda x: x[0])  # R²_adj の降順にソート

            pbar.update(1)  # 変数が追加された場合のみ進捗を更新

        pbar.close()  # tqdm を明示的に終了


    # top_n のモデルを抽出
    best_models = [model for _, model, _ in best_models_adj_r_squared]
    best_predictors_list = [predictors for _, _, predictors in best_models_adj_r_squared]

    # 結果の整理
    df_results_list = []

    for i, (_, model, selected_predictors) in enumerate(best_models_adj_r_squared):

        best_X = sm.add_constant(df[selected_predictors])
        predictions = model.predict(best_X)

        # モデルの結果をデータフレームに保存
        df_results = pd.DataFrame({
            "モデルID": f"Best_{i+1}",
            "発病率（実測値）": response,
            "発病率（予測値）": predictions,
            "誤差": response - predictions
        })

        # メタデータと選択された説明変数のみを結合
        df_results = pd.concat([meta_data, df_results, df[selected_predictors]], axis=1)
        df_results_list.append(df_results)

    # すべてのモデルの結果を統合
    df_final = pd.concat(df_results_list, axis=0, ignore_index=True).reset_index(drop=True)

    print("逐次選択法による線形重回帰分析が完了しました。")
    print(f"最良のモデルの決定係数 (R²_adj): {[round(x[0], 4) for x in best_models_adj_r_squared]}")
    print("---------------------------------------------------")

    return best_models, best_predictors_list, df_final
