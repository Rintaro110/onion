import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import os

from scipy.stats import pearsonr
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter


# --- period名の対応マップ（収穫日分析用） ---
period_map_by_col = {
    2: "貯蔵調査",
    3: "収穫日",
    4: "5月下旬",
    5: "5月上旬",
    6: "4月下旬",
    7: "4月上旬",
    8: "3月下旬",
    9: "3月上旬",
    10: "2月下旬",
    11: "2月上旬"
}

# ------------------------------
# 【メイン関数】
# ------------------------------
def analyze_and_export_correlation(df, target_type="cor1", output_path="results_it/correlation_output.xlsx"):
    print("---------------------------------------------------")
    print("相関分析を開始します。")
    print(f"target_type: {target_type}")
    print(f"出力先: {output_path}")

    if target_type == "cor1":
        analyze_weather_by_period(df, output_path)

    elif target_type == "cor2":
        analyze_harvest_vs_past(df, output_path, target_col="収穫日")

    elif target_type == "cor3":
        analyze_weather_overall(df, output_path)

    else:
        raise ValueError("target_type must be 'cor1', 'cor2', or 'cor3'")

    print(f"相関分析結果を {output_path} に保存しました。")
    print("---------------------------------------------------")



def compute_correlation_and_significance(subset, valid_cols, period, output_dir):
    """
    各気象変数と incidence との相関係数および p値を計算し、ヒートマップをPNGで保存し、
    結果をExcelに保存（p値が高いものは背景を赤色にして視認性を向上）
    """
    result = []
    for col in valid_cols:
        x = subset[col]
        y = subset["incidence"]
        x_y = pd.concat([x, y], axis=1).dropna()
        if len(x_y) < 3:
            continue
        r, p = pearsonr(x_y[col], x_y["incidence"])
        result.append({"変数": col, "相関係数": r, "p値": p})

    result_df = pd.DataFrame(result).set_index("変数")
    result_df = result_df.sort_values("相関係数", key=abs, ascending=False)

    # Excel出力
    excel_path = os.path.join(output_dir, f"corr_significance_{period}.xlsx")
    result_df.to_excel(excel_path, sheet_name="correlation")

    # 背景色つけ：p値が0.05以上 → 薄赤
    wb = load_workbook(excel_path)
    ws = wb["correlation"]
    red_fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")

    for row in range(2, ws.max_row + 1):
        p_val = ws[f"C{row}"].value
        if p_val >= 0.05:
            for col_letter in ["B", "C"]:
                ws[f"{col_letter}{row}"].fill = red_fill

    wb.save(excel_path)

    # ヒートマップ出力（可視性強化）
    plt.figure(figsize=(max(10, len(result_df) * 0.4), 4))
    sns.heatmap(result_df[["相関係数"]], annot=True, fmt=".2f", cmap="coolwarm", center=0,
                annot_kws={"size": 7}, linewidths=0.5, cbar=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(f"【{period}】発病率と気象データの相関")
    plt.tight_layout()

    img_path = os.path.join(output_dir, f"heatmap_{period}.png")
    plt.savefig(img_path)
    plt.close()



# ------------------------------
# 【1】weather_by_period：periodごとに気象 vs incidence
# ------------------------------
def analyze_weather_by_period(df, output_path):
    weather_cols = [col for col in df.columns if col.endswith("days") or any(col.endswith(f"_{d}days") for d in [7,14,30,60,90,120,150,180])]

    periods = df["period"].dropna().unique()
    output_dir = os.path.dirname(output_path)

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        any_written = False

        for period in periods:
            subset = df[df["period"] == period]

            valid_cols = [col for col in weather_cols if subset[col].notna().sum() > 0]
            available_cols = valid_cols + ["incidence"]

            if subset[available_cols].dropna().empty or len(valid_cols) == 0:
                print(f"[!] {period} は欠損が多くスキップされました")
                continue

            corr_matrix = subset[available_cols].corr()
            corr_subset = corr_matrix.loc[valid_cols, ["incidence"]]

            safe_sheet_name = str(period)[:30]
            corr_subset.to_excel(writer, sheet_name=safe_sheet_name)
            any_written = True

            # ヒートマップ保存 & 有意性評価
            compute_correlation_and_significance(subset, valid_cols, period, output_dir)

        if not any_written:
            pd.DataFrame({"注意": ["有効なデータが1つもありませんでした"]}).to_excel(writer, sheet_name="empty")


# ------------------------------
# 【2】harvest：収穫日 vs 過去の発病率（ピボットで整理）
# ------------------------------
def analyze_harvest_vs_past(df, output_path, target_col="収穫日"):
    df_pivot = df.pivot_table(index=["brand", "year"], columns="period", values="incidence")
    df_pivot = df_pivot.rename(columns=period_map_by_col)

    if target_col not in df_pivot.columns:
        raise ValueError(f"{target_col} の発病率データが見つかりません。period_map_by_colを確認してください。")

    target_idx = list(df_pivot.columns).index(target_col)
    pre_cols = list(df_pivot.columns[:target_idx])

    corr_matrix = df_pivot[pre_cols + [target_col]].corr()
    corr_subset = corr_matrix.loc[pre_cols, [target_col]]

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a" if os.path.exists(output_path) else "w") as writer:
        corr_subset.to_excel(writer, sheet_name=f"{target_col}_vs_past")

    plt.figure(figsize=(8, len(pre_cols) * 0.4 + 2))
    sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap="YlGnBu", center=0)
    plt.title(f"【{target_col} vs 過去発病率】相関係数ヒートマップ")
    plt.tight_layout()
    plt.show()


# ------------------------------
# 【3】weather：全体の気象 vs incidence（参考：削除も可）
# ------------------------------
def analyze_weather_overall(df, output_path):
    disease_cols = [col for col in df.columns if "発病" in col or "incidence" in col]
    weather_cols = [col for col in df.columns if col.endswith("days")]

    corr_matrix = df[disease_cols + weather_cols].corr()
    corr_subset = corr_matrix.loc[disease_cols, weather_cols]

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        corr_subset.to_excel(writer, sheet_name="weather_disease_corr")

    plt.figure(figsize=(18, len(disease_cols) * 0.5 + 4))
    sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("発病率と気象データの相関")
    plt.tight_layout()
    plt.show()
