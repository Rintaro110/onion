import pandas as pd
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.dates import DateFormatter
import seaborn as sns
import os

def import_disease_data_sammary(file_path, verbose=False):
    print("---------------------------------------------------")
    print(f"データ読み込み完了: {file_path}")
    

    excel = pd.ExcelFile(file_path)
    records = []

    for sheet in excel.sheet_names:
        try:
            year = int(sheet)
        except ValueError:
            continue
        
        # 対象行（0-based）
        target_row = 18 if year <= 2007 else 16

        # データをヘッダーなしで読み込む
        df = pd.read_excel(file_path, sheet_name=sheet, header=None)

        # 指定行のB列が対象の品種名か確認
        product = df.iat[target_row, 1]  # B列（index=1）
        if pd.isna(product) or product != target_name:
            continue  # このシートはスキップ

        # 日付は1行目のC列以降（index=2〜）
        date_headers = pd.to_datetime(df.iloc[0, 2:], errors='coerce')
        values = df.iloc[target_row, 2:]

        for date, val in zip(date_headers, values):
            if pd.isna(date):
                continue
            records.append({
                '年度': year,
                '品種名': product,
                '日付': date.strftime('%m-%d'),
                '発病率': val
            })

    # データフレーム化
    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(['年度', '日付']).reset_index(drop=True)

    # 保存
    os.makedirs("results_it", exist_ok=True)
    result_df.to_excel("results_it/時系列発病率データ.xlsx", index=False)

    if verbose:
        print(result_df.head())

    print("病害データ取得完了。")
    print("---------------------------------------------------")

    return result_df


def plot_disease_trends_by_year(excel_path, target_name="ターザン", output_dir=None, show=True):
    """
    Plot disease incidence trends by year for a specific variety, using MM-DD observation dates.

    Parameters:
    ----------
    excel_path : str
        Path to the Excel file containing the dataset.
    target_name : str
        Target variety name (default: "ターザン").
    output_dir : str or None
        Directory to save the plot. If None, the plot is not saved.
    show : bool
        If True, the plot is displayed (default: True).
    """

    df = pd.read_excel(excel_path)

    # Filter by target variety
    df = df[df["品種名"] == target_name].copy()
    if df.empty:
        print(f"No data found for variety '{target_name}'.")
        return

    # Convert MM-DD string to datetime (dummy year 2000)
    df['日付_dt'] = pd.to_datetime('2000-' + df['日付'], format='%Y-%m-%d')

    # Plot
    plt.figure(figsize=(10, 10))
    sns.lineplot(data=df, x='日付_dt', y='発病率', hue='年度', palette='tab20', marker='o')

    plt.title(f"Disease incidence trends of {target_name}", fontsize=14)
    plt.xlabel("Observation date (MM-DD)")
    plt.ylabel("Incidence rate")

    # Format x-axis to hide the year
    date_format = DateFormatter("%m-%d")
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.xticks(rotation=45)
    plt.legend(title="Year", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Save plot if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{target_name}_incidence_trends.png")
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


    

if __name__ == '__main__':


    file_path = "resources/disease_data/onion_disease_sammary.xlsx"
    target_name = "ターザン"

    # 病害データをインポート
    disease_data= import_disease_data_sammary(file_path)
    print(disease_data)

    plot_disease_trends_by_year(
        excel_path="時系列発病率データ.xlsx",
        target_name="ターザン",
        output_dir="results_it/disease_trends",
        show=False
    )
