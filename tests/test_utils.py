import pandas as pd
import numpy as np

import statsmodels.api as sm
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns
import psutil  # 追加: CPUコア数確認用
from functools import partial
from itertools import combinations
from statsmodels.stats.outliers_influence import variance_inflation_factor
from multiprocessing import Pool, cpu_count
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

import import_meteorological_data as md  # このファイルには、process_data_from_sumoto関数が含まれています
import import_desease_data as dd  # このファイルには、extract_data_from_directory関数が含まれています
import data_preprocessing as dp

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PowerTransformer
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import random
import statsmodels.api as sm



def calculate_vif(X):
    """VIFを計算し、10を超えるVIFがあるかどうかを返し、VIFが高い列のリストを返す。"""
    
    # 分散がほぼ0のカラムを除去
    X = X.loc[:, X.var() > 1e-10]

    # Xが空、または列が2つ未満の場合、VIF計算をスキップ
    if X.empty or len(X.columns) < 2:
        # print("Skipping VIF calculation: not enough data.")
        return True, []  # 問題なしと判断してTrueを返す
    
    # VIFスコアを計算
    vif_scores = []
    for i, column in enumerate(X.columns):
        try:
            r_squared_i = sm.OLS(X.values[:, i], X.values[:, np.arange(X.shape[1]) != i]).fit().rsquared
            if np.isinf(r_squared_i) or np.isnan(r_squared_i) or r_squared_i > 0.999:
                # print(f"Assigned infinite VIF due to high collinearity (r_squared_i near 1) in column '{column}'")
                vif = np.inf
            else:
                vif = 1. / (1. - r_squared_i)
            vif_scores.append((column, vif))
            # print(f"VIF for {column} is {vif}")
        except ZeroDivisionError as e:
            # print(f"ZeroDivisionError calculating VIF for {column}: {e}")
            vif_scores.append((column, np.inf))  # ゼロ除算エラーが発生した場合、無限大を割り当て
        except Exception as e:
            # print(f"Error calculating VIF for {column}: {e}")
            vif_scores.append((column, np.inf))  # その他のエラーが発生した場合も無限大を割り当て

    # VIFスコアから10を超えるものがあるかチェック
    high_vif_columns = []
    for column, vif in vif_scores:
        if vif > 10:
            # print(f"High VIF > 10 detected: {vif} in column '{column}'")
            high_vif_columns.append(column)

    # 高いVIFが検出された場合はFalseを返し、除外された変数のリストを表示
    if high_vif_columns:
        # print(f"The following columns were excluded due to high VIF: {high_vif_columns}")
        return False, high_vif_columns  # VIFが10を超えるものがあればFalseを返す

    # print("All VIF scores are below 10.")
    return True, []  # すべてのVIFが10未満であれば、Trueを返す

def perform_regression_analysis(df, max_k=None, exclude_strings=None, contains_strings=None):
    response = df['発病率']
    predictors = dp.filter_predictors(df, exclude_strings, contains_strings)

    if not predictors:
        raise ValueError("Filtered predictors are empty. Check exclude and include filters.")
    
    # フィルター後の予測変数を表示
    print(f"Filtered predictors: {predictors}")

    best_models_adj_r_squared = []
    excluded_combinations = set()  # 除外する変数の組み合わせを保持する集合

    total_combinations = sum(1 for _ in combinations(predictors, min(len(predictors), max_k if max_k is not None else len(predictors))))
    print(f"Total combinations to evaluate: {total_combinations}")

    processed_combinations = 0
    next_progress_update = 10

    for k in range(1, min(len(predictors), max_k if max_k is not None else len(predictors)) + 1):
        for combo in combinations(predictors, k):

            # 高いVIFの組み合わせを含むものは除外
            # excluded_combinations は組み合わせの集合である必要がある
            if any(set(combo).issubset(excluded_set) for excluded_set in excluded_combinations):
                print(f"Skipping combination due to high VIF exclusion: {combo}")
                continue

            processed_combinations += 1
            progress = (processed_combinations / total_combinations) * 100
            if progress >= next_progress_update:
                print(f"Progress: {int(progress)}%")
                next_progress_update += 10

            X = sm.add_constant(df[list(combo)])

            if X.shape[1] > 2:
                # VIFチェック
                valid_vif, high_vif_columns = calculate_vif(X.drop(columns='const'))
                if not valid_vif:
                    excluded_combinations.add(frozenset(high_vif_columns))  # VIFが高い変数を将来的に除外するリストに追加
                    continue

            # モデルのフィッティング
            model = sm.OLS(response, X).fit()
            
            # 調整R二乗値を基準に最良のモデルを保持
            if len(best_models_adj_r_squared) < 3 or model.rsquared_adj > best_models_adj_r_squared[-1].rsquared_adj:
                if len(best_models_adj_r_squared) == 3:
                    best_models_adj_r_squared.pop()
                best_models_adj_r_squared.append(model)
                best_models_adj_r_squared.sort(key=lambda x: x.rsquared_adj, reverse=True)

    best_models_adj_r_squared = best_models_adj_r_squared[:3]
    best_X_adj_r_squared = [sm.add_constant(df[list(model.model.exog_names[1:])]) for model in best_models_adj_r_squared]

    return best_models_adj_r_squared, best_X_adj_r_squared

def perform_regression_with_sequential_selection(df, labels, max_k=None, exclude_strings=None, contains_strings=None, seed=None):
    """
    必須変数 (labels) を常に含みながら逐次選択法を使用して最適なモデルを構築する。
    ランダム性を持たせるために説明変数の評価順をランダムに設定。

    Args:
        df (pd.DataFrame): データフレーム。
        labels (list): 必須の説明変数のリスト。
        max_k (int): 選択可能な説明変数の最大数。
        exclude_strings (list): 除外する変数名の部分文字列リスト。
        contains_strings (list): 含める変数名の部分文字列リスト。
        seed (int): ランダム性の固定用シード値。

    Returns:
        best_model (sm.OLS): 最適なモデル。
        best_predictors (list): 最適な説明変数のリスト。
    """
    response = df['発病率']
    predictors = dp.filter_predictors(df, exclude_strings, contains_strings)
    
    # 予測変数が空の場合は必須変数のみでモデルを構築
    if not predictors:
        print("No predictors found after filtering. Using only mandatory labels.")
        X = sm.add_constant(df[labels])
        model = sm.OLS(response, X).fit()
        return model, labels, X, response

    print(f"Filtered predictors: {predictors}")

    if seed is not None:
        random.seed(seed)

    selected_predictors = list(labels)
    remaining_predictors = list(predictors)
    best_model = None
    best_r_squared_adj = -float('inf')

    while remaining_predictors and (max_k is None or len(selected_predictors) - len(labels) < max_k):
        best_candidate = None
        random.shuffle(remaining_predictors)

        for predictor in remaining_predictors:
            current_predictors = selected_predictors + [predictor]
            X = sm.add_constant(df[current_predictors])

            # VIFチェック
            if X.shape[1] > 2:
                valid_vif, high_vif_columns = calculate_vif(X.drop(columns='const'))
                if not valid_vif:
                    continue

            # モデルフィッティング
            model = sm.OLS(response, X).fit()

            if model.rsquared_adj > best_r_squared_adj:
                best_r_squared_adj = model.rsquared_adj
                best_candidate = predictor
                best_model = model

        if best_candidate:
            selected_predictors.append(best_candidate)
            remaining_predictors.remove(best_candidate)
        else:
            break

    # もし最適なモデルが見つからなかった場合、必須変数のみでモデルを作成
    if best_model is None:
        print("No better model found. Using only mandatory labels.")
        X = sm.add_constant(df[labels])
        best_model = sm.OLS(response, X).fit()
        selected_predictors = labels

    return best_model, selected_predictors, X, response

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

def perform_regression_analysis_parallel_top3(df, max_k=None, exclude_strings=None, contains_strings=None, cpu_count=1, pca=False, correlation_threshold=0.8):
    """
    複数の組み合わせに対して並列で回帰分析を行い、最良のモデルを選定する関数。
    """
    response = df['発病率']

    if pca:
        df, predictors = dp.perform_pca_and_select_predictors(df, exclude_strings=exclude_strings, include_strings=contains_strings)
    else:
        predictors = dp.select_and_filter_predictors(df, correlation_threshold=correlation_threshold, exclude_strings=exclude_strings, include_strings=contains_strings)

    if not predictors:
        raise ValueError("Filtered predictors are empty. Check exclude and include filters.")
    
    print(f"Filtered predictors: {predictors}")

    # インデックスの整合性を確認
    response = response.loc[df.index]


    best_models_adj_r_squared = []

    print ("max_k: ", max_k)
    total_combinations = sum(1 for _ in combinations(predictors, min(len(predictors), max_k if max_k is not None else len(predictors))))
    print(f"Total combinations to evaluate: {total_combinations}")

    # 全ての組み合わせを取得
    all_combinations = [combo for k in range(1, min(len(predictors), max_k if max_k is not None else len(predictors)) + 1) 
                        for combo in combinations(predictors, k)]
    
    # 使用しているCPUコア数を表示
    print(f"Using {cpu_count} CPU cores for parallel processing.")
    
    # プロセスプールの設定
    with mp.Pool(processes=cpu_count) as pool:
        # tqdmを使って進捗バーを表示
        with tqdm(total=len(all_combinations), desc="Processing combinations") as pbar:
            for result in pool.imap_unordered(partial(process_combination, df=df, response=response), all_combinations):
                if result is not None:
                    model = result
                    if len(best_models_adj_r_squared) < 2 or model.rsquared_adj > best_models_adj_r_squared[-1].rsquared_adj:
                        if len(best_models_adj_r_squared) ==2:
                            best_models_adj_r_squared.pop()
                        best_models_adj_r_squared.append(model)
                        best_models_adj_r_squared.sort(key=lambda x: x.rsquared_adj, reverse=True)

                # 進捗バーを更新
                pbar.update(1)

    best_models_adj_r_squared = best_models_adj_r_squared[:3]
    best_X_adj_r_squared = [sm.add_constant(df[list(model.model.exog_names[1:])]) for model in best_models_adj_r_squared]

    return best_models_adj_r_squared, best_X_adj_r_squared

def perform_regression_analysis_parallel_top1(df, max_k=None, exclude_strings=None, contains_strings=None, cpu_count=1, pca=False, correlation_threshold=0.8):
    """
    複数の組み合わせに対して並列で回帰分析を行い、最良のモデルを選定する関数。
    """
    response = df['発病率']

    if pca:
        df, predictors = dp.perform_pca_and_select_predictors(df, exclude_strings=exclude_strings, include_strings=contains_strings)
    else:
        predictors = dp.select_and_filter_predictors(df, correlation_threshold=correlation_threshold, exclude_strings=exclude_strings, include_strings=contains_strings)

    if not predictors:
        raise ValueError("Filtered predictors are empty. Check exclude and include filters.")
    
    print(f"Filtered predictors: {predictors}")

    # インデックスの整合性を確認
    response = response.loc[df.index]

    best_model_adj_r_squared = None
    best_r_squared_adj = -float('inf')
    
    print("max_k: ", max_k)
    total_combinations = sum(1 for _ in combinations(predictors, min(len(predictors), max_k if max_k is not None else len(predictors))))
    print(f"Total combinations to evaluate: {total_combinations}")

    # 全ての組み合わせを取得
    all_combinations = [combo for k in range(1, min(len(predictors), max_k if max_k is not None else len(predictors)) + 1) 
                        for combo in combinations(predictors, k)]
    
    print(f"Using {cpu_count} CPU cores for parallel processing.")
    
    with mp.Pool(processes=cpu_count) as pool:
        with tqdm(total=len(all_combinations), desc="Processing combinations") as pbar:
            for result in pool.imap_unordered(partial(process_combination, df=df, response=response), all_combinations):
                if result is not None:
                    model = result
                    if model.rsquared_adj > best_r_squared_adj:
                        best_r_squared_adj = model.rsquared_adj
                        best_model_adj_r_squared = model
                pbar.update(1)

    if best_model_adj_r_squared is None:
        raise ValueError("No valid model was found.")

    best_X = sm.add_constant(df[list(best_model_adj_r_squared.model.exog_names[1:])])
    
    return best_model_adj_r_squared, list(best_model_adj_r_squared.model.exog_names[1:]), best_X, response

def perform_correlation_analysis_and_save_to_excel(df, exclude_strings=None, contains_strings=None, filename="correlation_analysis.xlsx"):
    """
    発病率と気象データの相関係数を計算し、エクセルファイルに書き出す。
    
    :param df: データフレーム
    :param exclude_strings: 除外するカラム名に部分一致する文字列のリスト
    :param contains_strings: 選択するカラム名に部分一致する文字列のリスト
    :param filename: 保存するエクセルファイルの名前
    """
    # 発病率と説明変数（気象データ）のフィルタリング
    response = df['発病率']
    predictors = dp.filter_predictors(df, exclude_strings, contains_strings)

    if not predictors:
        raise ValueError("Filtered predictors are empty. Check exclude and include filters.")
    
    # 発病率と各気象データの相関係数を計算
    correlation_results = {}
    for predictor in predictors:
        correlation = df[[response.name, predictor]].corr().iloc[0, 1]  # 発病率との相関
        correlation_results[predictor] = correlation

    # 結果をデータフレームに変換
    correlation_df = pd.DataFrame(list(correlation_results.items()), columns=['気象情報', '相関係数'])

    # 相関係数で降順にソート（絶対値ではなくそのまま）
    correlation_df = correlation_df.sort_values(by='相関係数', ascending=False)

    # 結果をエクセルファイルに保存
    correlation_df.to_excel(filename, index=False)
    print(f"相関分析結果を {filename} に保存しました。")
