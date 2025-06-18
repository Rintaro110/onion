import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import japanize_matplotlib
import re
from sklearn.linear_model import LinearRegression

# ファイル名サニタイズ
def sanitize_filename(s):
    return re.sub(r'[\\/:"*?<>|]+', '_', s)

# 結果保存：features列をカンマ区切りで保存
def save_results(train_results, test_results, train_all, test_all, output_dir="results"):

    df_final = save_sequential_results(train_results, test_results, train_all, test_all, output_excel=f"{output_dir}/sequential_results.xlsx")
    summary = evaluate_sequential_model(train_results, test_results)
    plot_train_scatter(df_final, output_dir=f"{output_dir}/plots")
    plot_test_lines(df_final, output_dir=f"{output_dir}/plots")
    plot_sequential_model_results(train_results, test_results, output_dir=f"{output_dir}/plots")

    print("✅ 逐次型モデルの結果保存完了")



def save_sequential_results(
    train_results, test_results, train_all, test_all, 
    output_excel="outputs/sequential_results.xlsx",
    target_col="log_incidence"
):

    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    all_rows = []

    # モデル情報カラムを追加（線形モデルの場合：係数・t値・p値を出力）
    train_results = train_results.copy()
    coef_list, tvalues_list, pvalues_list = [], [], []

    for idx, row in train_results.iterrows():
        period = row["period"]
        features = row["features"] if isinstance(row["features"], list) else [row["features"]]
        pred_col = f"{period}_pred"
        model = row.get("model", None)

        # train
        df_train_period = train_all[train_all["period"] == period].copy()
        df_train_period["data_type"] = "train"

        # test
        df_test_period = test_all[test_all["period"] == period].copy()
        df_test_period["data_type"] = "test"

        df_period = pd.concat([df_train_period, df_test_period])
        df_period["period"] = period

        # 共通カラムのみ抽出
        base_cols = ["brand", "year", "period", target_col, pred_col, "data_type"]
        cols_exist = [col for col in base_cols if col in df_period.columns]
        df_period = df_period[cols_exist].rename(columns={target_col: "actual", pred_col: "predicted"})

        all_rows.append(df_period)

        # --- モデル情報のテキスト化（線形モデルのみ、XGBoostは別途feature_importances等で）
        if model is not None and hasattr(model, "params"):
            # statsmodelsのRegressionResults
            coef = model.params.to_dict()
            tvals = model.tvalues.to_dict()
            pvals = model.pvalues.to_dict()
        elif model is not None and hasattr(model, "coef_"):
            # sklearnのLinearRegressionなど
            coef = {f: float(c) for f, c in zip(features, np.atleast_1d(model.coef_))}
            tvals = {}
            pvals = {}
        else:
            coef = {}
            tvals = {}
            pvals = {}

        coef_list.append(str(coef))
        tvalues_list.append(str(tvals))
        pvalues_list.append(str(pvals))

    df_final = pd.concat(all_rows, ignore_index=True)

    # features列をカンマ区切りの文字列として保存
    train_results["features"] = train_results["features"].apply(
        lambda x: ','.join(x) if isinstance(x, list) else str(x)
    )
    train_results["coef"] = coef_list
    train_results["tvalues"] = tvalues_list
    train_results["pvalues"] = pvalues_list

    with pd.ExcelWriter(output_excel) as writer:
        df_final.to_excel(writer, sheet_name="Predictions", index=False)
        train_results.to_excel(writer, sheet_name="Train_Summary", index=False)
        test_results.to_excel(writer, sheet_name="Test_Summary", index=False)

    print(f"✅ 予測結果Excel保存完了: {output_excel}")
    return df_final


# 全体性能サマリ
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

# periodごと実測・予測値推移プロット


def plot_train_scatter(df_final, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    d = df_final[df_final["data_type"] == "train"]
    periods = d["period"].unique()

    for period in periods:
        data = d[d["period"] == period]
        X = data["actual"].values.reshape(-1, 1)
        y = data["predicted"].values

        plt.figure(figsize=(6, 6))
        plt.scatter(X, y, alpha=0.7, label="train")
        if len(X) > 1:
            lr = LinearRegression().fit(X, y)
            xx = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            plt.plot(xx, lr.predict(xx), color="red", lw=2, label="回帰直線")
        plt.title(f"{period} (n={len(X)})")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # 各periodごとにファイル名を分けて保存
        fname = f"train_scatter_{sanitize_filename(str(period))}.pdf"
        save_path = os.path.join(output_dir, fname)
        plt.savefig(save_path)
        plt.close()
        print(f"✅ train散布図＋回帰線保存: {save_path}")


def plot_test_lines(
    df_final, output_dir="outputs/plots", target_col="log_incidence"
):
    os.makedirs(output_dir, exist_ok=True)
    d = df_final[df_final["data_type"] == "test"]
    years = sorted(d["year"].unique())
    periods = list(d["period"].unique())
    period_map = {p: i for i, p in enumerate(periods)}

    # 年度ごとに色を割り当て
    color_list = ["red", "blue", "green"]
    if len(years) > len(color_list):
        from matplotlib import cm
        colors = cm.get_cmap("tab10", len(years))
        color_list = [colors(i) for i in range(len(years))]

    plt.figure(figsize=(10, 7))
    for i, year in enumerate(years):
        df_year = d[d["year"] == year]
        df_year = df_year.sort_values("period", key=lambda x: x.map(period_map))
        color = color_list[i % len(color_list)]
        plt.plot(df_year["period"], df_year["actual"], marker="o", color=color, linestyle="-", label=f"{year} actual", alpha=0.8)
        plt.plot(df_year["period"], df_year["predicted"], marker="x", color=color, linestyle="--", label=f"{year} predicted", alpha=0.8)
    plt.xlabel("Period")
    plt.ylabel(target_col)
    plt.title(f"Test: 年ごと {target_col} の推移")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize="small", ncol=1)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "test_actual_vs_predicted_lines.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ test 年ごとの推移（実測・予測）を保存: {save_path}")


# 性能プロット
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
