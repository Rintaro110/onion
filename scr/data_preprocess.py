from datetime import timedelta
import pandas as pd
import datetime
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import japanize_matplotlib


import import_weatherdata as wd
import import_diseasedata as dd

def preprocess_data(
        disease_df, 
        weather_df, 
        period_order, 
        use_past_incidence=False, 
        remove_outliers=False,
        output_dir="results_it",
        test_years=None
    ):
    """
    前処理段階で学習データとテストデータを分割
    """
    print("---------------------------------------------------")
    print("データの前処理を開始します。")
    

    merged_df = merge_disease_and_weather(disease_df, weather_df)
    cleaned_df = drop_missing_records(merged_df)
    log_df = log_transform_target(cleaned_df, target_col="incidence", log_offset=1, new_col="log_incidence")
    if use_past_incidence:
        log_df_a = dd.add_log_incidence_cumsum(log_df, period_order)
        log_df_b = dd.add_all_log_incidence_diffs(log_df_a, period_order)
        log_df = dd.add_past_period_log_actuals(log_df_b, period_order)

    cleaned_log_df = detect_and_remove_outliers(log_df, period_order=period_order, target_col="log_incidence", output_dir=f"{output_dir}/boxplot", remove_outliers=remove_outliers)
    stepwise_data = pivot_by_period(cleaned_log_df, period_order)

    # 学習用・テスト用に分割
    if test_years is not None:
        print(f"学習データとテストデータを分割します。テスト年: {test_years}")
        train_df = stepwise_data[~stepwise_data["year"].isin(test_years)].reset_index(drop=True)
        test_df  = stepwise_data[stepwise_data["year"].isin(test_years)].reset_index(drop=True)

        # 保存処理
        save_merged_data_to_excel(stepwise_data, f"{output_dir}/merged_features.xlsx")
        print(f"✅ 学習データ: {train_df.shape}, テストデータ: {test_df.shape}")

        return train_df, test_df
    
    else:
        print("テスト年が指定されていないため、全データを学習用として扱います。")
        save_merged_data_to_excel(stepwise_data, f"{output_dir}/merged_features.xlsx")
        return stepwise_data
    

def merge_disease_and_weather(disease_df, weather_df):
    """
    Merge disease incidence data with weather features (average and sum)
    and add cumulative incidence and period-to-period difference columns.

    """
    print("---------------------------------------------------")
    print("病害データと気象データのマージを開始します。")


    # 気象データマージ
    all_records = []

    for _, row in disease_df.iterrows():
        year = row["year"]
        obs_date = row["date"]
        period = row["period"]
        record = row.to_dict()

        if obs_date is None or period is None:
            print(f"Skipped: year={year}, period={period}, obs_date={obs_date}")
            continue

        # 平均値
        weather_windows_avg = wd.get_multiple_weather_period(weather_df, year, obs_date, period)
        for period_key, avg_df in weather_windows_avg.items():
            for _, feat_row in avg_df.iterrows():
                item = feat_row["weather_item"]
                val = feat_row["value"]
                col_name = f"{item}_{period_key}"
                record[col_name] = val

        # 積算値（sum）
        weather_windows_sum = wd.get_multiple_weather_sum_period(weather_df, year, obs_date, period)
        for period_key, sum_df in weather_windows_sum.items():
            for _, feat_row in sum_df.iterrows():
                item = feat_row["weather_item"]
                val = feat_row["value"]
                col_name = f"{item}_{period_key}"
                record[col_name] = val

        all_records.append(record)

    merged_df = pd.DataFrame(all_records)

    print("病害データと気象データのマージが完了しました。")
    print("---------------------------------------------------")
    return merged_df

def drop_missing_records(df):
    """
    Display and remove rows with missing values in key columns.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame containing disease and weather features

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with rows containing missing values removed
    """
    print("---------------------------------------------------")
    print("欠損値を含む行を確認し、除外します。")
    
    # 欠損を確認する対象の重要列
    key_columns = [
        "brand", "year", "period", "date", "incidence",
    ]

    # 欠損を含む行を抽出
    missing_df = df[df[key_columns].isnull().any(axis=1)]

    # 欠損がある行を表示
    if not missing_df.empty:
        print("⚠️ 以下の行に欠損があります（除外されます）:")
        print(missing_df[["brand", "year", "period", "date", "incidence"]])
    else:
        print("✅ 欠損値は見つかりませんでした。")

    # 欠損を除去した DataFrame を返す
    cleaned_df = df.dropna(subset=key_columns).reset_index(drop=True)

    print("---------------------------------------------------")

    return cleaned_df

def log_transform_target(df, target_col="incidence", log_offset=1, new_col="log_incidence"):
    """
    目的変数を log(x + offset) で変換し、新たなカラムを target_col の隣に挿入して返す。
    """
    df = df.copy()
    log_values = np.log(df[target_col] + log_offset)

    # 挿入位置を取得
    target_index = df.columns.get_loc(target_col)
    
    # 挿入
    df.insert(loc=target_index + 1, column=new_col, value=log_values)

    return df

def detect_and_remove_outliers(
    df, 
    period_order,
    target_col="log_incidence", 
    output_dir="results_it/outlier",
    remove_outliers=True
):
    """
    log変換済みカラムに対応し、period_order順でループ
      - 各periodごとに箱ひげ図を作成
      - IQR法（上側のみ）で外れ値検出・除去（任意）

    Parameters
    ----------
    df : pd.DataFrame
    period_order : list
        periodの順序リスト
    target_col : str
        対象カラム名（log変換済みカラム等）
    output_dir : str
        グラフ保存先
    remove_outliers : bool
        Trueなら外れ値を除去して箱ひげ図を描画、Falseなら元データで描画する

    Returns
    -------
    pd.DataFrame
        外れ値除去済みまたは元データのコピー
    """

    os.makedirs(output_dir, exist_ok=True)
    cleaned_df = df.copy()
    outlier_years = set()

    for period in period_order:
        sub = df[df["period"] == period]
        if sub.empty:
            continue

        # 1. IQR法（上側のみ）で外れ値を検出（削除するかは remove_outliers で判定）
        vals = sub[target_col].dropna()
        if len(vals) < 2:
            print(f"[{period}] データ数不足で外れ値判定をスキップしました。")
            plot_df = sub
        else:
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            outlier_mask = (sub[target_col] > upper)
            outliers = sub[outlier_mask]

            if not outliers.empty:
                print(f"[{period}] log-IQR外れ値として検出された年度:")
                for _, row in outliers.iterrows():
                    print(f"  year={row['year']} (brand={row['brand']}, {target_col}={row[target_col]:.3g})")
                    outlier_years.update([(row['year'], row['brand'], period)])
            else:
                print(f"[{period}] 外れ値は見つかりませんでした。")

            # 外れ値を除くかどうかに応じて使用するデータを決定
            if remove_outliers:
                plot_df = sub[~outlier_mask]
                # 全体のクリーンデータからも除去
                for _, row in outliers.iterrows():
                    mask = (
                        (cleaned_df["year"] == row["year"]) &
                        (cleaned_df["brand"] == row["brand"]) &
                        (cleaned_df["period"] == period)
                    )
                    cleaned_df = cleaned_df[~mask]
            else:
                plot_df = sub

        # 2. 箱ひげ図の描画（除外あり／なしに応じて）
        plt.figure(figsize=(2.8, 4.2))
        sns.boxplot(y=target_col, data=plot_df, color="lightblue", width=0.5)
        plt.title(f"{period}", fontsize=18)
        plt.ylabel(target_col, fontsize=18)
        plt.xlabel("")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout(pad=0.2)

        suffix = "_removed" if remove_outliers else "_raw"
        plt.savefig(os.path.join(output_dir, f"{period}_boxplot{suffix}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"===> 外れ値除外後のデータサイズ: {cleaned_df.shape}")
    return cleaned_df

def pivot_by_period(df, period_order):
    print("---------------------------------------------------")
    print("🔧 pivot_by_period(): 整形＋NaN列除去処理を開始します")

    df = df.copy()
    # periodをカテゴリ型にして順序を明示
    df["period"] = pd.Categorical(df["period"], categories=period_order, ordered=True)
    # ブランド→period_order→yearの順で並べる
    df = df.sort_values(["brand", "period", "year"]).reset_index(drop=True)

    na_cols = df.columns[df.isna().all()].tolist()
    cleaned_df = df.drop(columns=na_cols)

    print(f"✅ 完成データサイズ: {cleaned_df.shape}")
    if na_cols:
        print("🧹 以下の列は全行NaNのため削除されました:")
        for col in na_cols:
            print(f"  - {col}")
    else:
        print("✅ 全ての列に有効なデータが存在しています。")
    print("---------------------------------------------------")
    return cleaned_df

def save_merged_data_to_excel(df, output_path):
    """
    Save the merged disease and weather DataFrame to an Excel file.

    Parameters
    ----------
    df : pd.DataFrame
        The merged DataFrame from merge_disease_and_weather
    output_path : str
        File path to save the Excel file (e.g., 'results_it/merged_features.xlsx')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"✅ Merged data saved to: {output_path}")


if __name__ == "__main__":
    # 観測日を指定
    wd_file_path = "resources/meteorological_data/1990_2025_sumoto.xlsx"
    dd_file_path = "resources/disease_data/onion_disease_sammary.xlsx"
    
    output_dir = "results_it"
    target_name = "ターザン"
    start_year = 1994
    end_year  = 2023

    period_order = [
        "2月上旬", "2月下旬", "3月上旬", "3月下旬", "4月上旬", "4月下旬", "5月上旬", "5月下旬", "収穫日", "貯蔵調査"
    ]

    # 病害データをインポート
    disease_list= dd.import_disease_data(dd_file_path, target_name, start_year, end_year)
    disease_df = pd.DataFrame(disease_list)

    # データ抽出
    weather_df = wd.extract_meteorological_data(wd_file_path, start_year, end_year)
    # データマージ
    merged_data = preprocess_data(disease_df, weather_df, period_order)
    print(merged_data)
    

