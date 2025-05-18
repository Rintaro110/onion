import pandas as pd
from openpyxl import load_workbook
import os

def save_to_excel(df, output_path):
    """気象項目を列に展開し、Excel形式で保存"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ピボット変換：気象項目を列に
    wide_df = df.pivot_table(index=["年度", "日付"], columns="気象項目", values="気象データ", aggfunc="first").reset_index()

    # 気象項目が列ラベルになっているのでフラット化（列名の階層を解除）
    wide_df.columns.name = None

    # Excel保存
    wide_df.to_excel(output_path, index=False)
    print(f"気象データを保存しました: {output_path}")


def extract_meteorological_data(file_path, start_year, end_year):
    print("---------------------------------------------------")
    print(f"気象データを読み込み中: {file_path}")
    print(f"対象年度: {start_year}年 ～ {end_year}年")

    wb = load_workbook(file_path, data_only=True)
    records = []

    for target_year in range(start_year, end_year + 1):
        matched_sheet = None
        for sheet_name in wb.sheetnames:
            try:
                sheet_start = int(sheet_name.split("-")[0])
                sheet_end = int(sheet_name.split("-")[1])
            except:
                continue
            if sheet_start <= target_year <= sheet_end:
                matched_sheet = sheet_name
                break

        if not matched_sheet:
            print(f"⚠️ 年度 {target_year} を含むシートが見つかりません。スキップします。")
            continue

        ws = wb[matched_sheet]

        # 2行目から気象項目名を取得（B列以降）
        item_names = [cell.value for cell in ws[2]][1:]

        # データ行の処理
        for row in ws.iter_rows(min_row=3, values_only=True):
            raw_date = row[0]
            if raw_date is None:
                continue

            try:
                date = pd.to_datetime(raw_date).date()  # 時刻を除去
            except:
                continue

            # 年度判定（12月は翌年）
            year = date.year + 1 if date.month == 12 else date.year
            if year != target_year:
                continue

            for i, value in enumerate(row[1:], start=0):
                item = item_names[i]
                records.append({
                    "気象項目": item,
                    "年度": year,
                    "日付": date,
                    "気象データ": value
                })
        print(f"{target_year} 年度のデータを取得しました。")
    print("全ての年度のデータを取得しました。")

    df = pd.DataFrame(records)

    os.makedirs("results_it", exist_ok=True)
    output_path = os.path.join("results_it", f"{start_year}to{end_year}_weather_data.xlsx")
    save_to_excel(df, output_path)

    print("気象データ取得完了。")
    print("---------------------------------------------------")

    return df

if __name__ == "__main__":
    file_path = "resources/meteorological_data/1990_2025_sumoto.xlsx"
    df = extract_meteorological_data(file_path, 1998, 2002)