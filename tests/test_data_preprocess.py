import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler    
from sklearn.preprocessing import StandardScaler    
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.decomposition import PCA


def merge_datasets(desease_data, meteorological_data, use_average_only=False, detect_outliers=False, z_threshold=3):
    # 結果を格納するリスト
    merged_data = []
    print("データ結合処理を開始します。")

    # 各品種と年度についてループ
    for variety, years_data in desease_data.items():
        for year, locations_data in years_data.items():
            
            # 年度に対応するすべての気象データラベルのデータを取得
            year_meteorological_data = {}
            periods_exist = False

            for label, years in meteorological_data.items():
                if year in years:
                    for month, periods_or_value in years[year].items():
                        if isinstance(periods_or_value, dict):
                            # periodsが存在する場合の処理
                            periods_exist = True
                            for period in ['上旬', '中旬', '下旬']:
                                if period in periods_or_value:
                                    year_meteorological_data[f"{label}_{month}_{period}"] = periods_or_value[period]
                        else:
                            # periodsが存在しない場合の処理
                            year_meteorological_data[f"{label}_{month}"] = periods_or_value

            if not year_meteorological_data:
                print(f"    警告: 年度 {year} の気象データが見つかりません。")
                continue

            # 各場所とそのデータについてループ
            for location, incidence_rate in locations_data:
                
                if use_average_only:
                    # 平均のみを使用する場合、場所が平均でない場合はスキップ
                    if location != '平均':
                        continue

                # 結合データのレコードを作成
                record = {
                    "品種": variety,
                    "年度": year,
                    "場所": location,
                    "発病率": incidence_rate
                }

                # 月別気象データと平均をレコードに追加
                record.update(year_meteorological_data)

                # リストにレコードを追加
                merged_data.append(record)

    # データフレームに変換
    df = pd.DataFrame(merged_data)
    print("データフレーム変換完了。")
    print(f"DataFrame shape: {df.shape}")
    
    # 50%以上の欠損値がある列を削除
    threshold = len(df) * 0.5 if periods_exist else len(df) * 0.2
    df = df.loc[:, df.isnull().sum() <= threshold]
    print(f"DataFrame shape after dropping columns with more than {threshold} missing values: {df.shape}")

    # 欠損データの数を表示
    missing_data = df.isnull().sum()
    print(f"Missing data count per column:\n{missing_data}")

    # 欠損値がある行を事前に保持
    rows_with_missing_data = df[df.isnull().any(axis=1)].copy()
    print(f"削除される欠損値を含む行の数: {len(rows_with_missing_data)}")
    
    if not rows_with_missing_data.empty:
        print("削除される欠損値を含む行の詳細:")
        print(rows_with_missing_data)

    # 欠損データの処理方法（欠損値を持つ行を削除する）
    df.dropna(inplace=True)
    print(f"DataFrame shape after dropna: {df.shape}")


    # 外れ値検出を行う場合
    if detect_outliers:
        # 発病率列の平均と標準偏差を計算
        mean_incidence = df["発病率"].mean()
        std_incidence = df["発病率"].std()

        # 片側Zスコア法で外れ値を検出
        outliers = df[df["発病率"] > mean_incidence + z_threshold * std_incidence]
        print(f"除外される外れ値の行の数: {len(outliers)}")
        
        # 除外されたデータの年度、場所、発病率を出力
        if not outliers.empty:
            print("除外される外れ値の行の詳細（年度、場所、発病率）:")
            print(outliers[["年度", "場所", "発病率"]])

        # 外れ値を除外
        df = df[~df.index.isin(outliers.index)]
        print(f"DataFrame shape after outlier removal: {df.shape}")

    # 発病率が6以上の行を削除
    # df = df[df["発病率"] < 6]

    # 最終データフレームの形状を表示
    print(f"最終的なDataFrameの形状: {df.shape}")

    # 最終データフレームを返す
    return df

def filter_predictors(df, exclude_strings=None, include_strings=None):
    """
    特定の条件でカラムを除外または選択する。

    :param df: DataFrame
    :param exclude_strings: 除外するカラム名に部分一致する文字列のリスト
    :param include_strings: 選択するカラム名に部分一致する文字列のリスト
    :return: フィルタリング後のカラムリスト
    """

    # 基本除外ラベルを先にフィルタリング
    predictors = [col for col in df.columns if col not in ['発病率', '品種', '年度', '場所']]
    
    
    if exclude_strings:
        # exclude_stringsに部分一致するものを除外
        predictors = [col for col in predictors if not any(ex_string in col for ex_string in exclude_strings)]
    
    if include_strings:
        # include_stringsに部分一致するもののみを選択
        filtered_predictors = []
        for col in predictors:
            if any(in_string in col for in_string in include_strings):
                filtered_predictors.append(col)
        predictors = filtered_predictors

    return predictors

def normalize_data(df):
    """
    正規化を行う関数。
    データフレーム内の全ての数値列を0から1の範囲にスケーリングする。

    Args:
        df (pd.DataFrame): データフレーム。

    Returns:
        pd.DataFrame: 正規化後のデータフレーム。
    """
    # 数値列のみを対象にフィルタリング
    numeric_columns = df.select_dtypes(include=['number']).columns

    # 除外するラベルを設定し、数値列から除外
    exclude_columns = ['発病率', '品種', '年度', '場所']
    target_columns = [col for col in numeric_columns if col not in exclude_columns]

    # MinMaxScalerを適用
    scaler = MinMaxScaler()
    df[target_columns] = scaler.fit_transform(df[target_columns])

    print("すべての対象列の正規化を実行しました。")
    return df

def standardize_data(df):
    """
    標準化を行う関数。
    指定された列の値を平均0、標準偏差1にスケーリングする。

    Args:
        df (pd.DataFrame): データフレーム。
        columns (list): 標準化する列のリスト。

    Returns:
        pd.DataFrame: 標準化後のデータフレーム。
    """


   # 数値列のみを対象にフィルタリング
    numeric_columns = df.select_dtypes(include=['number']).columns

    # 基本除外ラベルを先にフィルタリング
    exclude_columns = ['発病率', '品種', '年度', '場所']
    target_columns = [col for col in numeric_columns if col not in exclude_columns]

    # 標準化を実行

    scaler = StandardScaler()
    df[target_columns] = scaler.fit_transform(df[target_columns])

    print("標準化を実行しました。")
    return df

def select_and_filter_predictors(df, target_variable='発病率', correlation_threshold=0.8, exclude_strings=None, include_strings=None, output_file="high_correlation_pairs.xlsx"):
    """
    説明変数の相関行列と文字列条件に基づいて、特定の説明変数を選択し、相関が高いペアを
    目的変数との相関に基づいて絞り込む関数。高相関ペアをエクセルに出力する。

    Parameters:
        df (DataFrame): 説明変数と目的変数を含むデータフレーム
        target_variable (str): 目的変数の列名（デフォルトは "発病率"）
        correlation_threshold (float): 説明変数の相関係数のしきい値（デフォルトは0.8）
        exclude_strings (list, optional): 除外するカラム名に部分一致する文字列のリスト
        include_strings (list, optional): 選択するカラム名に部分一致する文字列のリスト
        output_file (str): 高相関ペアを出力するエクセルファイルのパス（デフォルトは"high_correlation_pairs.xlsx"）

    Returns:
        selected_predictors (list): 絞り込まれた説明変数のリスト
    """

    # 特定の条件に基づいてカラムをフィルタリング
    predictors = [col for col in df.columns if col not in [target_variable, '品種', '年度', '場所']]
    
    # 除外する文字列が指定されている場合、該当するカラムを除外
    if exclude_strings:
        predictors = [col for col in predictors if not any(ex_string in col for ex_string in exclude_strings)]
    
    # 選択する文字列が指定されている場合、該当するカラムのみを選択
    if include_strings:
        filtered_predictors = [col for col in predictors if any(in_string in col for in_string in include_strings)]
        predictors = filtered_predictors

    # 目的変数を除いた説明変数のデータフレーム
    predictors_df = df[predictors].select_dtypes(include=[float, int])

    # 説明変数間の相関行列を計算
    correlation_matrix = predictors_df.corr()

    # 高相関の変数ペアを格納するためのリスト
    high_correlation_pairs = []

    # 相関行列から高相関ペアを抽出
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= correlation_threshold:
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                high_correlation_pairs.append((var1, var2, corr_value))

    print("高相関ペア:")
    for pair in high_correlation_pairs:
        print(f"{pair[0]} と {pair[1]} の相関: {pair[2]}")

    # 高相関ペアをエクセルに出力
    if high_correlation_pairs:
        high_corr_df = pd.DataFrame(high_correlation_pairs, columns=['Variable 1', 'Variable 2', 'Correlation'])
        
        # Excelファイルに書き込む
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            high_corr_df.to_excel(writer, sheet_name="High Correlation Pairs", index=False)
        print(f"高相関ペアが {output_file} に保存されました。")

    # 目的変数と説明変数の相関係数を計算
    target_correlation = df[[target_variable] + list(predictors_df.columns)].corr()[target_variable].drop(target_variable)

    # 高相関ペアから、目的変数との相関がより高い変数を残す
    variables_to_drop = set()
    for var1, var2, _ in high_correlation_pairs:
        if abs(target_correlation[var1]) >= abs(target_correlation[var2]):
            variables_to_drop.add(var2)  # var1の相関が高ければvar2を削除
        else:
            variables_to_drop.add(var1)  # var2の相関が高ければvar1を削除

    # 残す変数リストを作成
    selected_predictors = [var for var in predictors_df.columns if var not in variables_to_drop]

    print("\n選択された説明変数:")
    print(selected_predictors)

    return selected_predictors

def perform_pca_and_select_predictors(df, target_variable='発病率', exclude_strings=None, include_strings=None, n_components=None, output_file="pca_results.xlsx"):
    """
    説明変数の選択とPCAによる次元削減を行い、主成分をエクセルに出力し、主成分を説明変数として返す関数。

    Parameters:
        df (DataFrame): 説明変数と目的変数を含むデータフレーム
        target_variable (str): 目的変数の列名（デフォルトは "発病率"）
        exclude_strings (list, optional): 除外するカラム名に部分一致する文字列のリスト
        include_strings (list, optional): 選択するカラム名に部分一致する文字列のリスト
        n_components (int, optional): PCAの主成分数（デフォルトはNoneで、分散の80%以上を説明するよう設定）
        output_file (str): PCAの結果を出力するエクセルファイルのパス（デフォルトは "pca_results.xlsx"）

    Returns:
        df (DataFrame): 元のデータフレームに主成分を追加したもの
        selected_predictors (list): 主成分の名前をリストとして返す
    """
    # 特定の条件に基づいてカラムをフィルタリング
    predictors = [col for col in df.columns if col not in [target_variable, '品種', '年度', '場所']]
    
    if exclude_strings:
        predictors = [col for col in predictors if not any(ex_string in col for ex_string in exclude_strings)]
    
    if include_strings:
        predictors = [col for col in predictors if any(in_string in col for in_string in include_strings)]

    # 説明変数のデータフレームを作成（数値型のカラムのみ）
    predictors_df = df[predictors].select_dtypes(include=[float, int])

    if predictors_df.empty:
        raise ValueError("No valid predictors after filtering. Check exclude/include conditions and data.")

    # 標準化を適用
    scaler = StandardScaler()
    predictors_scaled = scaler.fit_transform(predictors_df)

    # PCAを実行
    pca = PCA(n_components=n_components if n_components else 0.9)
    pca_transformed = pca.fit_transform(predictors_scaled)
    pca_df = pd.DataFrame(pca_transformed, columns=[f'PC{i+1}' for i in range(pca_transformed.shape[1])], index=predictors_df.index)

    # 主成分の分散比率、累積分散比率、主成分負荷行列を計算
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    loading_matrix = pd.DataFrame(pca.components_, columns=predictors_df.columns, index=[f'PC{i+1}' for i in range(len(pca.components_))])

    print("\n主成分の分散比率:", explained_variance_ratio)
    print("累積分散比率:", cumulative_variance_ratio)
    print("\n主成分負荷行列:\n", loading_matrix)

    # 主成分を元のデータフレームに追加
    df = pd.concat([df, pca_df], axis=1)

    # 主成分名をリストとして返す
    selected_predictors = list(pca_df.columns)

    print("\n選択された説明変数 (主成分):")
    print(selected_predictors)

    # PCA結果をエクセルに出力
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 主成分データフレームを保存
        pca_df.to_excel(writer, sheet_name="PCA Components", index=True)

        # 分散比率を保存
        variance_df = pd.DataFrame({
            "Principal Component": [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
            "Explained Variance Ratio": explained_variance_ratio,
            "Cumulative Explained Variance": cumulative_variance_ratio
        })
        variance_df.to_excel(writer, sheet_name="Explained Variance", index=False)

        # 主成分負荷行列を保存
        loading_matrix.to_excel(writer, sheet_name="Loading Matrix", index=True)

    print(f"PCAの結果が {output_file} に保存されました。")

    return df, selected_predictors

def filter_predictors_by_variance_and_correlation(df, target_variable='発病率', exclude_strings=None, include_strings=None, variance_threshold=0.01, correlation_threshold=0.2, output_file="filtered_predictors.xlsx"):
    """
    各説明変数の分散分析と目的変数との相関分析を行い、分散が小さく、かつ目的変数との相関が小さい変数を除外する関数。

    Parameters:
        df (DataFrame): 説明変数と目的変数を含むデータフレーム
        target_variable (str): 目的変数の列名（デフォルトは "発病率"）
        variance_threshold (float): 分散のしきい値（この値以下の分散の変数は除外）
        correlation_threshold (float): 相関係数のしきい値（この値以下の相関の変数は除外）
        output_file (str): フィルタリング結果を出力するエクセルファイルのパス（デフォルトは "filtered_predictors.xlsx"）

    Returns:
        filtered_df (DataFrame): フィルタリング後のデータフレーム
        removed_predictors (list): 除外された説明変数のリスト
    """
    # 特定の条件に基づいてカラムをフィルタリング
    predictors = [col for col in df.columns if col not in [target_variable, '品種', '年度', '場所']]
    
    # 除外する文字列が指定されている場合、該当するカラムを除外
    if exclude_strings:
        predictors = [col for col in predictors if not any(ex_string in col for ex_string in exclude_strings)]
    
    # 選択する文字列が指定されている場合、該当するカラムのみを選択
    if include_strings:
        predictors = [col for col in predictors if any(in_string in col for in_string in include_strings)]

    # 目的変数を除いた説明変数のデータフレーム
    predictors_df = df[predictors].select_dtypes(include=[float, int])
    
    # 各変数の分散を計算
    variances = predictors_df.var()
    
    # 目的変数との相関を計算
    correlations = predictors_df.corrwith(df[target_variable])
    
    # 分散が基準値以下、かつ目的変数との相関がしきい値以下の変数を選択
    removed_predictors = variances[(variances <= variance_threshold) & (correlations.abs() <= correlation_threshold)].index.tolist()
    
    # 上記条件を満たさない変数（残す変数）を選択
    filtered_predictors = [col for col in predictors if col not in removed_predictors]

    # 結果をデータフレームで保存
    filtered_df = df[filtered_predictors + [target_variable]]
    
    # 取り除かれた変数の分散と相関を取得
    removed_variances = variances.loc[removed_predictors]
    removed_correlations = correlations.loc[removed_predictors]

    # 結果をエクセルに保存
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 各変数の分散を保存
        variance_df = pd.DataFrame({
            "Predictor": variances.index,
            "Variance": variances.values
        }).sort_values(by="Variance", ascending=False)
        variance_df.to_excel(writer, sheet_name="Variance Analysis", index=False)
        
        # 各変数と目的変数の相関を保存
        correlation_df = pd.DataFrame({
            "Predictor": correlations.index,
            "Correlation with Target": correlations.values
        }).sort_values(by="Correlation with Target", ascending=False)
        correlation_df.to_excel(writer, sheet_name="Correlation Analysis", index=False)
        
        # フィルタリングされた変数を保存
        filtered_predictors_df = pd.DataFrame({
            "Predictor": filtered_predictors
        })
        filtered_predictors_df.to_excel(writer, sheet_name="Filtered Predictors", index=False)
        
        # 除外された変数の分散と相関を保存
        removed_predictors_df = pd.DataFrame({
            "Removed Predictor": removed_predictors,
            "Variance": removed_variances.values,
            "Correlation with Target": removed_correlations.values
        })
        removed_predictors_df.to_excel(writer, sheet_name="Removed Predictors", index=False)
        
    print(f"フィルタリング結果が {output_file} に保存されました。")
    print(f"除外された説明変数とその詳細:")
    print(removed_predictors_df)
    
    return filtered_df




