import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler    
from sklearn.preprocessing import StandardScaler    
from sklearn.decomposition import PCA


def merge_datasets(desease_data, meteorological_data, use_average_only=False):
    """
    発病率データと気象データを統合する．

    Args:
        desease_data (dict): 発病率データの辞書
        meteorological_data (dict): 気象データの辞書
        use_average_only (bool): 平均のみを使用するかどうか 
    
    Returns:
        pd.DataFrame: 統合されたデータフレーム
    """

    merged_data = []
    print("データ結合処理を開始します。")

    for variety, years_data in desease_data.items():
        for year, locations_data in years_data.items():
            # 年度に対応するすべての気象データラベルのデータを取得
            year_meteorological_data = {}
            periods_exist = False

            for label, years in meteorological_data.items():
                if year in years:
                    for month, periods_or_value in years[year].items():
                        if isinstance(periods_or_value, dict):
                            periods_exist = True
                            for period in ['上旬', '中旬', '下旬']:
                                if period in periods_or_value:
                                    year_meteorological_data[f"{label}_{month}_{period}"] = periods_or_value[period]
                        else:
                            year_meteorological_data[f"{label}_{month}"] = periods_or_value

            if not year_meteorological_data:
                print(f"warnign!: 年度 {year} の気象データが見つかりません。")
                continue

            # 各場所とそのデータについてループ
            for location, incidence_rate in locations_data:
                if use_average_only and location != 'avg':
                    continue  # 平均のみを使用する場合、平均でないデータはスキップ

                # 結合データのレコードを作成
                record = {
                    "品種": variety,
                    "年度": year,
                    "場所": location,
                    "発病率": incidence_rate
                }

                # 月別気象データと平均をレコードに追加
                record.update(year_meteorological_data)
                merged_data.append(record)

    # データフレームに変換
    df = pd.DataFrame(merged_data)
    print(f"データフレーム変換完了: {df.shape}")
    
    return df

def drop_missing_values(df, threshold_ratio=0.3, verbose=True):
    """
    指定された閾値（割合）以上の欠損値を持つ列を削除し、欠損データを持つ行を削除する

    Args:
        df (pd.DataFrame): 欠損処理を行うデータフレーム
        threshold_ratio (float): 欠損値の許容割合（例: 0.3 → 30%以上の欠損がある列を削除）
        verbose (bool): True の場合、処理の詳細を出力

    Returns:
        pd.DataFrame: 欠損処理後のデータフレーム
    """
    # 30%以上の欠損値がある列を削除
    missing_threshold = len(df) * threshold_ratio
    df_cleaned = df.loc[:, df.isnull().sum() <= missing_threshold]

    if verbose:
        print(f"DataFrame shape after dropping columns with >{threshold_ratio * 100}% missing values: {df_cleaned.shape}")

    # 欠損データのカウント
    missing_counts = df_cleaned.isnull().sum()
    if verbose:
        print(f"欠損データ数（カラムごと）:\n{missing_counts}")

    # 欠損値を持つ行の特定
    missing_rows = df_cleaned[df_cleaned.isnull().any(axis=1)]

    if verbose and not missing_rows.empty:
        print(f"削除予定の欠損データ行数: {len(missing_rows)}")
        print("削除される欠損データの詳細:")
        print(missing_rows)

    else: 
        print("欠損データはありません。")

    # 欠損値を持つ行を削除
    df_cleaned.dropna(inplace=True)

    if verbose:
        print(f"DataFrame shape after dropna: {df_cleaned.shape}")

    return df_cleaned

def detect_and_remove_outliers(df, target_variable='発病率', z_threshold=3):
    """
    外れ値を検出し、指定されたZスコアのしきい値を超える外れ値を除外する関数。

    Args:
        df (pd.DataFrame): データフレーム。
        target_variable (str): 目的変数の列名。
        z_threshold (int): Zスコアのしきい値。

    Returns:
        pd.DataFrame: 外れ値を除外したデータフレーム。
    """
    # 目的変数の平均と標準偏差を計算
    mean_value = df[target_variable].mean()
    std_value = df[target_variable].std()

    # 片側Zスコア法で外れ値を検出
    outliers = df[df[target_variable] > mean_value + z_threshold * std_value]
    print(f"除外される外れ値の行の数: {len(outliers)}")
    
    # 除外されたデータの詳細を出力
    if not outliers.empty:
        print("除外される外れ値の行の詳細（年度、場所、発病率）:")
        print(outliers[["年度", "場所", target_variable]])

    # 外れ値を除外
    df_cleaned = df[~df.index.isin(outliers.index)]
    print(f"DataFrame shape after outlier removal: {df_cleaned.shape}")

    return df_cleaned

def save_dataframe_to_csv(df, filename="merged_data.csv", index=False, verbose=True):
    """
    データフレームをCSVファイルに保存する汎用関数

    Args:
        df (pd.DataFrame): 保存するデータフレーム
        filename (str): 保存するCSVファイルの名前（デフォルト: "merged_datasets.csv"）
        index (bool): インデックスを保存するか（デフォルト: False）
        verbose (bool): 処理の詳細を表示するか（デフォルト: True）

    Returns:
        None
    """
    try:
        df.to_csv(filename, index=index, encoding='utf-8-sig')
        if verbose:
            print(f"Success:データフレームをCSVとして保存しました: {filename}")
            print(f"最終的なデータフレームの形状: {df.shape}")
    except Exception as e:
        print(f"Error CSVの保存に失敗しました: {e}")

def filter_predictors(df, exclude_strings=None, include_strings=None):
    """
    特定の条件でカラムを除外または選択し、基本ラベル ['品種', '年度', '場所', '発病率'] を先頭に配置する。

    :param df: DataFrame
    :param exclude_strings: 除外するカラム名に部分一致する文字列のリスト
    :param include_strings: 選択するカラム名に部分一致する文字列のリスト
    :return: フィルタリング後のDataFrame
    """

    # 必ず含める基本ラベル（順序を維持）
    base_labels = ['品種', '年度', '場所', '発病率']
    
    # データフレーム内に存在する基本ラベルのみを取得
    existing_base_labels = [col for col in base_labels if col in df.columns]

    # すべてのカラムを取得
    predictors = df.columns.tolist()

    # 除外リストの適用
    if exclude_strings:
        predictors = [col for col in predictors if not any(ex_string in col for ex_string in exclude_strings)]
        print(f"除外されたカラム: {exclude_strings}")
    else:
        print("除外されたカラムはありません。")

    # 選択リストの適用
    if include_strings:
        predictors = [col for col in predictors if any(in_string in col for in_string in include_strings)]
        print(f"選択されたカラム: {include_strings}")
    else:
        print("選択されたカラムはありません。")

    # 残りのカラム（基本ラベルを除外したもの）
    other_columns = [col for col in predictors if col not in existing_base_labels]

    # 新しいカラム順（基本ラベルを先頭に配置）
    final_predictors = existing_base_labels + other_columns

    return df[final_predictors]  # フィルタリング後のDataFrameを返す

def preprocess_data(
        desease_data, 
        meteorological_data,
        use_average_only=False, 
        detect_outliers=False, 
        z_threshold=3,
        threshold_ratio=0.3,
        exclude_strings=None, 
        include_strings=None,
        output_file="outputs/merged_data.csv"):
    
    """ 
    発病率データと気象データを統合し、データクレンジングを行う

    Args:
        desease_data (dict): 発病率データの辞書
        meteorological_data (dict): 気象データの辞書
        use_average_only (bool): 平均のみを使用するかどうか
        detect_outliers (bool): 外れ値を検出して除外するかどうか
        z_threshold (int): Zスコアのしきい値    
        threshold_ratio (float): 欠損値の許容割合
        exclude_strings (list): 除外するカラム名に部分一致する文字列のリスト
        include_strings (list): 選択するカラム名に部分一致する文字列のリスト
        output_file (str): 出力ファイル名
    
    Returns:
        pd.DataFrame: 統合されたデータフレーム
    """

    # データ結合
    print("---------------------------------------------------")
    megrged_df = merge_datasets(desease_data, meteorological_data, use_average_only = use_average_only)
    
    # 欠損値処理
    print("---------------------------------------------------")
    dropped_df = drop_missing_values(megrged_df, threshold_ratio = threshold_ratio, verbose=True)
    
    # 外れ値検出＆除外
    print("---------------------------------------------------")
    if detect_outliers:
        outliered_df = detect_and_remove_outliers(dropped_df, target_variable = "発病率", z_threshold = z_threshold)

    # 説明変数の事前選択
    print("---------------------------------------------------")
    result_df = filter_predictors(outliered_df, exclude_strings=exclude_strings, include_strings=include_strings)
    
    # データフレームをCSVに保存
    print("---------------------------------------------------")
    save_dataframe_to_csv(result_df, filename=output_file)

    print(f"最終的なDataFrameの形状: {result_df.shape}")
    print("---------------------------------------------------")

    return result_df

