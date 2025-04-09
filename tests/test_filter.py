import pandas as pd

def filter_predictors(df, 
                      exclude_strings=None, 
                      include_strings=None, 
                      apply_correlation_filter=False, 
                      target_variable='発病率', 
                      correlation_threshold=0.8, 
                      filename="outputs/high_correlation_pairs.xlsx"):
    """
    特定の条件でカラムを除外または選択し、基本ラベル ['品種', '年度', '場所', '発病率'] を先頭に配置する。
    オプションとして、説明変数間の相関が高いペアを削除する機能も追加。

    Args:
        df (pd.DataFrame): データフレーム
        exclude_strings (list, optional): 除外するカラム名に部分一致する文字列のリスト
        include_strings (list, optional): 選択するカラム名に部分一致する文字列のリスト
        apply_correlation_filter (bool, optional): 高相関ペアの説明変数を削除するかどうか（デフォルト: False）
        target_variable (str, optional): 目的変数の列名（デフォルト: "発病率"）
        correlation_threshold (float, optional): 高相関と判定するしきい値（デフォルト: 0.8）
        output_file (str, optional): 高相関ペアを保存するエクセルファイル（デフォルト: "high_correlation_pairs.xlsx"）

    Returns:
        pd.DataFrame: フィルタリング後のDataFrame
    """

    # 必ず含める基本ラベル（順序を維持）
    base_labels = ['品種', '年度', '場所', '発病率']
    existing_base_labels = [col for col in base_labels if col in df.columns]
    
    # 説明変数候補の取得
    predictors = [col for col in df.columns if col not in existing_base_labels]

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

    # **高相関ペアを削除するオプションを適用**
    if apply_correlation_filter:
        print("高相関ペアのフィルタリングを適用します...")
        print("---------------------------------------------------")
        predictors = select_and_filter_predictors(df[[target_variable] + predictors], 
                                                  target_variable=target_variable, 
                                                  correlation_threshold=correlation_threshold, 
                                                  filename=filename)
    
    # 基本ラベル + 選択された説明変数を維持
    final_columns = existing_base_labels + predictors
    return df[final_columns]  # フィルタリング後のDataFrameを返す


def select_and_filter_predictors(df, target_variable='発病率', correlation_threshold=0.8, filename="outputs/high_correlation_pairs.xlsx"):
    """
    説明変数の相関行列と文字列条件に基づいて、特定の説明変数を選択し、相関が高いペアを
    目的変数との相関に基づいて絞り込む関数。高相関ペアをエクセルに出力する。

    Args:
        df (pd.DataFrame): 説明変数と目的変数を含むデータフレーム
        target_variable (str): 目的変数の列名（デフォルトは "発病率"）
        correlation_threshold (float): 説明変数の相関係数のしきい値（デフォルトは0.8）
        output_file (str): 高相関ペアを出力するエクセルファイルのパス（デフォルトは"high_correlation_pairs.xlsx"）

    Returns:
        selected_predictors (list): 絞り込まれた説明変数のリスト
    """

    # 説明変数のみ取得
    predictors = [col for col in df.columns if col != target_variable]
    
    # 数値型の説明変数を抽出
    predictors_df = df[predictors].select_dtypes(include=[float, int])

    # 相関行列を計算
    correlation_matrix = predictors_df.corr()

    # 高相関の変数ペアを抽出
    high_correlation_pairs = [
        (col1, col2, correlation_matrix.loc[col1, col2])
        for i, col1 in enumerate(correlation_matrix.columns)
        for j, col2 in enumerate(correlation_matrix.columns)
        if i < j and abs(correlation_matrix.loc[col1, col2]) >= correlation_threshold
    ]

    # 高相関ペアをエクセルに出力
    if high_correlation_pairs:
        high_corr_df = pd.DataFrame(high_correlation_pairs, columns=['Variable 1', 'Variable 2', 'Correlation'])
        high_corr_df.to_excel(filename, sheet_name="High Correlation Pairs", index=False)
        print(f"高相関ペアが {filename} に保存されました。")

    # 目的変数との相関を計算し、高相関ペアから1つを削除
    target_correlation = df[[target_variable] + predictors_df.columns.tolist()].corr()[target_variable].drop(target_variable)

    variables_to_drop = set()
    for var1, var2, _ in high_correlation_pairs:
        if abs(target_correlation[var1]) >= abs(target_correlation[var2]):
            variables_to_drop.add(var2)
        else:
            variables_to_drop.add(var1)

    # 残す変数リスト
    selected_predictors = [var for var in predictors_df.columns if var not in variables_to_drop]

    print("\n選択された説明変数:")
    print(selected_predictors)

    return selected_predictors