import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
import re

# ファイル名サニタイズ
def sanitize_filename(s):
    return re.sub(r'[\\/:"*?<>|]+', '_', s)

# ✅ 完全統合版：予測結果保存
def save_sequential_results(train_results, test_results, train_all, test_all, output_excel="outputs/sequential_results.xlsx"):
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)

    # ① 全予測データまとめ (train + test 合体)
    all_rows = []

    for idx, row in train_results.iterrows():
        period = row["period"]
        features = row["features"]
        pred_col = f"{period}_pred"

        # train
        df_train_period = train_all[train_all["period"] == period].copy()
        df_train_period["data_type"] = "train"

        # test
        df_test_period = test_all[test_all["period"] == period].copy()
        df_test_period["data_type"] = "test"

        df_period = pd.concat([df_train_period, df_test_period])
        df_period["period"] = period

        # 共通カラムのみ抽出
        base_cols = ["brand", "year", "period", "incidence", pred_col, "data_type"]
        cols_exist = [col for col in base_cols if col in df_period.columns]
        df_period = df_period[cols_exist].rename(columns={"incidence": "actual", pred_col: "predicted"})

        all_rows.append(df_period)

    df_final = pd.concat(all_rows, ignore_index=True)

    # ② 保存
    with pd.ExcelWriter(output_excel) as writer:
        df_final.to_excel(writer, sheet_name="Predictions", index=False)
        train_results.to_excel(writer, sheet_name="Train_Summary", index=False)
        test_results.to_excel(writer, sheet_name="Test_Summary", index=False)

    print(f"✅ 予測結果Excel保存完了: {output_excel}")
    return df_final

# ✅ 全体評価用：平均性能などを確認しやすく
def evaluate_sequential_model(train_results, test_results):
    summary = {
        "平均_Train_R2_adj": train_results["r2_adj"].mean(),
        "平均_Train_RMSE": train_results["rmse"].mean(),
        "平均_Test_RMSE": test_results["rmse"].mean(),
    }
    print("✅ モデル全体性能サマリ:")
    for k, v in summary.items():
        print(f"{k}: {v:.3f}")
    return summary

# ✅ プロット機能（散布図ではなく全体傾向のみ、軽量化版）
def plot_sequential_model_results(train_results, test_results, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    plt.plot(train_results["period"], train_results["r2_adj"], marker="o", label="Train R2_adj")
    plt.plot(train_results["period"], train_results["rmse"], marker="s", label="Train RMSE")
    plt.plot(test_results["period"], test_results["rmse"], marker="x", label="Test RMSE")

    plt.xticks(rotation=45)
    plt.xlabel("Period")
    plt.ylabel("Score")
    plt.title("逐次型モデルの性能推移")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "sequential_model_performance.pdf")
    plt.savefig(save_path, format="pdf")
    plt.close()
    print(f"✅ プロット保存: {save_path}")
