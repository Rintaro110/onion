import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import os
from scipy import stats

from scipy.stats import pearsonr
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter


# --- period名の対応マップ（収穫日分析用） ---


# ------------------------------
# 【メイン関数】
# ------------------------------
def analyze_and_export_correlation(df, target_type="cor1", target_col="log_incidence", output_path="results_it/correlation_output.xlsx"):
    print("---------------------------------------------------")
    print("相関分析を開始します。")
    print(f"target_type: {target_type}")
    print(f"出力先: {output_path}")

    if target_type == "cor1":
        analyze_weather_by_period(df, target_col,output_path)
        export_period_weather_corr_matrix(df, target_col, os.path.splitext(output_path)[0])

    elif target_type == "cor2":
        analyze_features_by_period(df, target_col, output_path)
        export_period_feature_corr_matrix(df, target_col, os.path.splitext(output_path)[0])
    elif target_type == "cor3":
        analyze_weather_overall(df, output_path)

    else:
        raise ValueError("target_type must be 'cor1', 'cor2', or 'cor3'")

    print(f"相関分析結果を {output_path} に保存しました。")
    print("---------------------------------------------------")



import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def compute_correlation_and_significance(subset, target_col, valid_cols, period, output_dir, scatter_threshold=0.4):
    """
    気象特徴量 vs incidence の
    1. 相関係数ヒートマップ（特徴量縦, incidence横）
    2. 有意性(p値)とともにランキング表をExcelに保存
    3. 強相関のみ散布図
    """
    os.makedirs(output_dir, exist_ok=True)
    corr_results = []
    pval_results = []

    # 1. 各特徴量とincidenceの相関＆p値計算
    for feat in valid_cols:
        # 欠損除去
        x = subset[feat]
        y = subset[target_col]
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            corr = np.nan
            pval = np.nan
        else:
            corr, pval = stats.pearsonr(x[mask], y[mask])
        corr_results.append((feat, corr))
        pval_results.append((feat, pval))

    # DataFrameに整形
    corr_df = pd.DataFrame(corr_results, columns=["feature", f"corr_with_{target_col}"])
    pval_df = pd.DataFrame(pval_results, columns=["feature", "p_value"])

    # ランキング：絶対値降順
    ranking_df = corr_df.copy()
    ranking_df["abs_corr"] = np.abs(ranking_df[f"corr_with_{target_col}"])
    ranking_df = ranking_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    ranking_df = ranking_df.drop(columns=["abs_corr"])
    # p値を右に
    ranking_df = ranking_df.merge(pval_df, on="feature")

    # 1. ランキングをExcelへ
    excel_path = os.path.join(output_dir, f"{period}_corr_ranking.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        ranking_df.to_excel(writer, sheet_name="ranking", index=False)
    print(f"  - 相関係数ランキング保存: {excel_path}")

    # 2. ヒートマップ（特徴量縦, incidence横）
    """ plt.figure(figsize=(10, len(valid_cols) * 0.35 + 1.5))
    # DataFrame: index=feature, columns=["incidence"]
    heatmap_data = ranking_df.set_index("feature")[[f"corr_with_{target_col}"]]
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", center=0, cbar=False)
    plt.title(f"{period} Correlation\n(feature vs {target_col})")
    plt.ylabel("Feature")
    plt.xlabel(target_col)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, f"{period}_corr_heatmap.pdf")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"  - 相関係数ヒートマップ保存: {heatmap_path}") """

    # 3. p値有意性
    signif_df = ranking_df[ranking_df["p_value"] < 0.05]
    if not signif_df.empty:
        print(f"  - 有意な相関（p<0.05）:")
        for _, row in signif_df.iterrows():
            print(f"    {row['feature']}: p={row['p_value']:.3g}, corr={row[f'corr_with_{target_col}']:.2f}")
    else:
        print("  - 有意な相関はありません（p<0.05）")

    # 4. 強相関散布図（しきい値以上のみ）
    strong = ranking_df[np.abs(ranking_df[f"corr_with_{target_col}"]) >= scatter_threshold]
    for _, row in strong.iterrows():
        feat = row['feature']
        plt.figure(figsize=(5, 4))
        sns.regplot(data=subset, x=feat, y=target_col, line_kws={'color': 'red'})
        plt.title(f"{period}: {feat} vs {target_col}\n(corr={row[f'corr_with_{target_col}']:.2f})")
        plt.xlabel(feat)
        plt.ylabel(target_col)
        plt.tight_layout()
        img_path = os.path.join(output_dir, f"{period}_{feat}_scatter.pdf")
        plt.savefig(img_path)
        plt.close()
        print(f"    - 強相関: {feat} の散布図保存: {img_path}")






# ------------------------------
# 【1】weather_by_period：periodごとに気象 vs incidence
# ------------------------------


def analyze_weather_by_period(df, target_col, output_path):
    # 気象特徴量列名は "_avg" または "_sum" で終わるものに限定
    weather_cols = [col for col in df.columns if col.endswith("_avg") or col.endswith("_sum")]

    periods = df["period"].dropna().unique()
    output_dir = os.path.dirname(output_path)

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        any_written = False

        for period in periods:
            print("---------------------------------------------------")
            print(f"Processing period: {period}")
            subset = df[df["period"] == period].copy()
            # そのperiodで有効なweather colsのみ（1つでも値があるものだけ）
            valid_cols = [col for col in weather_cols if subset[col].notna().sum() > 0]
            # 存在しないweather colを検出（全行NaN or カラム自体ない）
            removed_cols = [col for col in weather_cols if col not in valid_cols]

            available_cols = valid_cols + [target_col]
            if subset[available_cols].dropna().empty or len(valid_cols) == 0:
                print(f"[!] {period} は欠損が多くスキップされました")
                continue

            # 削除されたカラムを表示
            if removed_cols:
                print(f"Period: {period} - 削除されたカラム: {removed_cols}")

            # 相関行列
            corr_matrix = subset[available_cols].corr()
            corr_subset = corr_matrix.loc[valid_cols, [target_col]]

            safe_sheet_name = str(period)[:30]
            corr_subset.to_excel(writer, sheet_name=safe_sheet_name)
            any_written = True

            # ヒートマップ保存 & 有意性評価（関数は外部実装と仮定）
            compute_correlation_and_significance(subset, target_col, valid_cols, period, output_dir)

        if not any_written:
            pd.DataFrame({"Warning": ["No valid data found."]}).to_excel(writer, sheet_name="empty")

def export_period_weather_corr_matrix(df, target_col, output_path_prefix):
    """
    横軸：weather特徴量（_avg, _sumで分割して2つのファイルを作成）
    縦軸：period
    各セル：periodごとの気象特徴量とtarget_colの相関係数
    """
    periods = df["period"].dropna().unique()
    avg_cols = [col for col in df.columns if col.endswith("_avg")]
    sum_cols = [col for col in df.columns if col.endswith("_sum")]
    outdict = {}

    for suffix, weather_cols in [("_avg", avg_cols), ("_sum", sum_cols)]:
        corr_rows = []
        period_labels = []
        for period in periods:
            subset = df[df["period"] == period].copy()
            valid_cols = [col for col in weather_cols if subset[col].notna().sum() > 0]
            if not valid_cols or subset[valid_cols + [target_col]].dropna().empty:
                continue
            corrs = subset[valid_cols + [target_col]].corr()[target_col].loc[valid_cols]
            corr_rows.append(corrs)
            period_labels.append(period)
        # データフレーム化
        if corr_rows:
            corr_df = pd.DataFrame(corr_rows, index=period_labels)
            outdict[suffix] = corr_df

    # Excel出力
    with pd.ExcelWriter(f"{output_path_prefix}_period_weather_corr.xlsx", engine="openpyxl") as writer:
        for suffix, corr_df in outdict.items():
            sheet = "Weather" + suffix
            corr_df.to_excel(writer, sheet_name=sheet)
    print("横=weather, 縦=periodの相関行列をエクセル出力しました。")

    # Latex用にto_latexも出力（必要に応じて行・列数調整や四捨五入推奨）
    for suffix, corr_df in outdict.items():
        latex_str = corr_df.round(2).to_latex(
            index=True,
            caption=f"Pearson correlation between weather features ({suffix}) and {target_col} for each period.",
            label=f"tab:period_weather_corr_{suffix}",
            column_format="l" + "c" * corr_df.shape[1]
        )
        with open(f"{output_path_prefix}_corr_{suffix}.tex", "w", encoding="utf8") as f:
            f.write(latex_str)
        print(f"LaTeX表を{output_path_prefix}_corr_{suffix}.texに出力しました。")


# ------------------------------
# 【2】harvest：収穫日 vs 過去の発病率（ピボットで整理）
# ------------------------------
def analyze_features_by_period(df, target_col, output_path):
    """
    periodごとに、気象列・メタ情報以外の特徴量とtarget_colの相関を取る
    """
    # メタカラム・気象カラム以外を抽出
    meta_cols = [
        "brand", "year", "period", "date",
        "incidence", "log_incidence", "days_from_jan1"
    ]
    weather_cols = [col for col in df.columns if col.endswith("_avg") or col.endswith("_sum")]
    # diff, cumsum などは「相関分析したい説明変数」なら除外しない
    candidate_cols = [
        col for col in df.columns
        if col not in meta_cols and col not in weather_cols and col != target_col
    ]

    periods = df["period"].dropna().unique()
    output_dir = os.path.dirname(output_path)

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        any_written = False

        for period in periods:
            print("---------------------------------------------------")
            print(f"Processing period: {period}")
            subset = df[df["period"] == period].copy()

            # 1つでも値が入っている列のみ
            valid_cols = [col for col in candidate_cols if subset[col].notna().sum() > 0]
            removed_cols = [col for col in candidate_cols if col not in valid_cols]

            available_cols = valid_cols + [target_col]
            if subset[available_cols].dropna().empty or len(valid_cols) == 0:
                print(f"[!] {period} は欠損が多くスキップされました")
                continue

            # 削除されたカラムを表示
            if removed_cols:
                print(f"Period: {period} - 削除されたカラム: {removed_cols}")

            # 相関行列
            corr_matrix = subset[available_cols].corr()
            corr_subset = corr_matrix.loc[valid_cols, [target_col]]

            safe_sheet_name = str(period)[:30]
            corr_subset.to_excel(writer, sheet_name=safe_sheet_name)
            any_written = True

            # ヒートマップ保存 & 有意性評価（関数は外部実装と仮定）
            compute_correlation_and_significance(subset, target_col, valid_cols, period, output_dir)

        if not any_written:
            pd.DataFrame({"Warning": ["No valid data found."]}).to_excel(writer, sheet_name="empty")

def export_period_feature_corr_matrix(df, target_col, output_path_prefix):
    """
    横軸：各種特徴量（気象特徴量・メタ情報以外）
    縦軸：period
    各セル：periodごとの特徴量とtarget_colの相関係数
    """
    meta_cols = [
        "brand", "year", "period", "date",
        "incidence", "log_incidence", "days_from_jan1"
    ]
    weather_cols = [col for col in df.columns if col.endswith("_avg") or col.endswith("_sum")]
    candidate_cols = [
        col for col in df.columns
        if col not in meta_cols and col not in weather_cols and col != target_col
    ]
    periods = df["period"].dropna().unique()
    corr_rows = []
    period_labels = []

    for period in periods:
        subset = df[df["period"] == period].copy()
        valid_cols = [col for col in candidate_cols if subset[col].notna().sum() > 0]
        if not valid_cols or subset[valid_cols + [target_col]].dropna().empty:
            continue
        corrs = subset[valid_cols + [target_col]].corr()[target_col].loc[valid_cols]
        corr_rows.append(corrs)
        period_labels.append(period)

    # DataFrame化
    if corr_rows:
        corr_df = pd.DataFrame(corr_rows, index=period_labels)
        # Excel出力
        with pd.ExcelWriter(f"{output_path_prefix}_period_feature_corr.xlsx", engine="openpyxl") as writer:
            corr_df.to_excel(writer, sheet_name="Features")
        print("横=feature, 縦=periodの相関行列をエクセル出力しました。")

        # Latex用にto_latexも出力
        latex_str = corr_df.round(2).to_latex(
            index=True,
            caption=f"Pearson correlation between features and {target_col} for each period.",
            label=f"tab:period_feature_corr",
            column_format="l" + "c" * corr_df.shape[1]
        )
        with open(f"{output_path_prefix}_feature_corr.tex", "w", encoding="utf8") as f:
            f.write(latex_str)
        print(f"LaTeX表を{output_path_prefix}_feature_corr.texに出力しました。")
    else:
        print("有効なデータがありませんでした。")



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
