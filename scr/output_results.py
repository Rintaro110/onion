import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import japanize_matplotlib

def sanitize_filename(s):
    return re.sub(r'[\\/:"*?<>|]+', '_', s)

def save_sequential_regression_results(results_df, df_all, output_excel="outputs/sequential_results.xlsx"):
    """
    各 period の回帰結果（予測値、実測値、説明変数など）を Excel に保存する。

    Args:
        results_df (pd.DataFrame): sequential_linear_regression の返り値（period, r2_adj, rmse, features）
        df_all (pd.DataFrame): 各期のデータ全体（predictionカラムを含む）
        output_excel (str): 出力ファイル名

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)

    all_rows = []
    for idx, row in results_df.iterrows():
        period = row["period"]
        features = list(row["features"])
        pred_col = f"{period}_pred"

        df_period = df_all[df_all["period"] == period].copy()
        if pred_col not in df_period.columns:
            continue

        df_result = df_period[["brand", "year", "incidence"] + features].copy()
        df_result["predicted"] = df_period[pred_col]
        df_result["period"] = period
        df_result.rename(columns={"incidence": "actual"}, inplace=True)
        all_rows.append(df_result)

    if not all_rows:
        print("❌ 保存対象の結果が存在しません")
        return

    df_final = pd.concat(all_rows, ignore_index=True)
    df_final = df_final[["brand", "year", "period", "actual", "predicted"] + [col for col in df_final.columns if col not in ["brand", "year", "period", "actual", "predicted"]]]

    df_final.to_excel(output_excel, index=False)
    print(f"📁 回帰結果を保存しました: {output_excel}")

def plot_sequential_regression_results(df, folder="outputs", filename_prefix="sequential_fit"):
    """
    各 period ごとの実測値と予測値の散布図をプロット・保存します。

    Parameters
    ----------
    df : pd.DataFrame
        各 period ごとの回帰結果（実測値、予測値、period を含む）
    folder : str
        画像の保存先フォルダ
    filename_prefix : str
        保存ファイル名の接頭語
    """
    os.makedirs(folder, exist_ok=True)
    periods = df['period'].unique()

    for period in periods:
        sub_df = df[df['period'] == period]
        if sub_df.empty:
            continue

        plt.figure(figsize=(8, 6))
        plt.scatter(sub_df["actual"], sub_df["predicted"], alpha=0.7, edgecolors='k')
        plt.plot([sub_df["actual"].min(), sub_df["actual"].max()],
                 [sub_df["actual"].min(), sub_df["actual"].max()],
                 'r--', label="Ideal fit")

        plt.xlabel("Actual incidence")
        plt.ylabel("Predicted incidence")
        plt.title(f"{period} - Actual vs Predicted")
        plt.legend()
        plt.grid(True)

        safe_period = period.replace("/", "_").replace(" ", "_")
        file_path = os.path.join(folder, f"{filename_prefix}_{safe_period}.pdf")
        plt.savefig(file_path, format="pdf")
        plt.close()

    print(f"✅ プロットを保存しました: {folder}")

