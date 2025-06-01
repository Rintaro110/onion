from datetime import timedelta
import pandas as pd
import datetime
import os 

import import_weatherdata as wd
import import_diseasedata as dd


def merge_disease_and_weather(disease_df, weather_df):
    """
    Merge disease incidence data with weather features based on period-specific time windows.

    Parameters
    ----------
    disease_df : pd.DataFrame
        Disease incidence data (columns: brand, year, date, period, incidence)
    weather_df : pd.DataFrame
        Long-format weather data (columns: weather_item, year, date, value)
    save_path : str
        Path to save the merged Excel file

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with disease and weather features
    """

    print("---------------------------------------------------")
    print("病害データと気象データのマージを開始します。")

    all_records = []

    for _, row in disease_df.iterrows():
        year = row["year"]
        obs_date = row["date"]
        period = row["period"]  # <- 必須
        record = row.to_dict()

        # 欠損チェック（dateまたはperiodが欠けていたらスキップ）
        if obs_date is None or period is None:
            print(f"Skipped: year={year}, period={period}, obs_date={obs_date}")
            continue

        # period に応じて day_list を取得
        weather_windows = wd.get_multiple_weather_period(weather_df, year, obs_date, period)

        # 各期間・気象項目を展開
        for period_key, avg_df in weather_windows.items():
            for _, feat_row in avg_df.iterrows():
                item = feat_row["weather_item"]
                val = feat_row["value"]
                col_name = f"{item}_{period_key}"
                record[col_name] = val

        all_records.append(record)

    merged_df = pd.DataFrame(all_records)


    # 保存処理

    print("病害データと気象データのマージが完了しました。")
    print("---------------------------------------------------")
    return merged_df


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
        "平均気温(℃)_7days", "平均湿度(％)_7days", "平均蒸気圧(hPa)_7days",
        "平均風速(m/s)_7days", "日照時間(時間)_7days", "最低気温(℃)_7days",
        "最大風速(m/s)_7days", "最高気温(℃)_7days", "降水量の合計(mm)_7days"
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

def preprocess_data(disease_df, weather_df, period_order, use_past_incidence=False, save_path="results_it/merged_features.xlsx"):
    """
    Preprocess the data by merging disease and weather data.

    Parameters
    ----------
    disease_df : pd.DataFrame
        Disease incidence data (columns: brand, year, date, period, incidence)
    weather_df : pd.DataFrame
        Long-format weather data (columns: weather_item, year, date, value)
    save_path : str
        Path to save the merged Excel file

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with disease and weather features
    """
    merged_df = merge_disease_and_weather(disease_df, weather_df)
    cleaned_df = drop_missing_records(merged_df)
    stepwise_data = prepare_stepwise_prediction_data(cleaned_df, period_order, use_past_incidence)

    save_merged_data_to_excel(stepwise_data, save_path)


    return stepwise_data

    merge_disease_and_weather(disease_df, weather_df, save_path)


import pandas as pd

def prepare_stepwise_prediction_data(df, period_order, use_past_incidence=False):
    print("---------------------------------------------------")
    print("🔧 prepare_stepwise_prediction_data(): 整形＋NaN列除去処理を開始します")

    df_list = []
    for i, target_period in enumerate(period_order):
        current_df = df[df["period"] == target_period].copy()
        if current_df.empty:
            print(f"⚠️ {target_period} のデータが存在しません。スキップ。")
            continue

        if use_past_incidence:
            for past_period in period_order[:i]:
                col_name = f"{past_period}_incidence"
                past_data = df[df["period"] == past_period][["brand", "year", "incidence"]].rename(columns={"incidence": col_name})
                current_df = current_df.merge(past_data, on=["brand", "year"], how="left")

        df_list.append(current_df)

    if not df_list:
        print("❌ 全てのperiodでデータが見つからなかったため、処理を中断します。")
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)

    na_cols = full_df.columns[full_df.isna().all()].tolist()
    cleaned_df = full_df.drop(columns=na_cols)

    print(f"✅ 完成データサイズ: {cleaned_df.shape}")
    if na_cols:
        print("🧹 以下の列は全行NaNのため削除されました:")
        for col in na_cols:
            print(f"  - {col}")
    else:
        print("✅ 全ての列に有効なデータが存在しています。")

    print("---------------------------------------------------")
    
    return cleaned_df





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
    