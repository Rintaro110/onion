import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import japanize_matplotlib
import os

import data_preprocess as dp
import import_weatherdata as wd
import import_diseasedata as dd

# 1. STL分解
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import os

def analyze_stl_from_days(df, output_dir="results_it", filename="stl_from_days.png"):
    os.makedirs(output_dir, exist_ok=True)

    # 年ごとに集約された、days_from_jan1ベースの incidence を平均
    daily_mean = df.dropna(subset=["incidence", "days_from_jan1"]) \
                   .groupby("days_from_jan1")["incidence"].mean()

    # 時系列としてSTL分解（年内観測は概ね10回程度）
    stl = STL(daily_mean, period=5)  # periodは試行調整可
    result = stl.fit()

    fig = result.plot()
    fig.suptitle("STL Decomposition (Avg. by days_from_jan1)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

# 分析関数２：ACF/PACF
def analyze_acf_pacf_from_days(df, output_dir="results_it", filename="acf_pacf_from_days.png", lags=10):
    os.makedirs(output_dir, exist_ok=True)

    daily_mean = df.dropna(subset=["incidence", "days_from_jan1"]) \
                   .groupby("days_from_jan1")["incidence"].mean()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(daily_mean, lags=lags, ax=axes[0])
    plot_pacf(daily_mean, lags=lags, ax=axes[1])
    axes[0].set_title("ACF: incidence vs days_from_jan1")
    axes[1].set_title("PACF: incidence vs days_from_jan1")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# サンプル実行は環境依存のためここでは行わず、関数提供のみに留める。

if __name__ == "__main__":
    # 観測日を指定
    wd_file_path = "resources/meteorological_data/1990_2025_sumoto.xlsx"
    dd_file_path = "resources/disease_data/onion_disease_sammary.xlsx"
    
    output_dir = "results_it"
    target_name = "ターザン"
    start_year = 1994
    end_year  = 2023

    # 病害データをインポート
    disease_list= dd.import_disease_data(dd_file_path, target_name, start_year, end_year)
    disease_df = pd.DataFrame(disease_list)

    # データ抽出
    # weather_df = wd.extract_meteorological_data(wd_file_path, start_year, end_year)
    # データマージ
    # merged_data = pd.preprocess_data(disease_df, weather_df)

    # STL分解
    # plot_stl_decomposition(disease_df, period='貯蔵調査', output_dir=output_dir)
    analyze_stl_from_days(disease_df, output_dir=output_dir)
    # 自己相関・偏自己相関
    analyze_acf_pacf_from_days(disease_df, lags=8, output_dir=output_dir)